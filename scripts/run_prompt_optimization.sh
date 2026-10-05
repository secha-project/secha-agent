#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# Edit these defaults, or override them with the matching flags below.
# Qwen solves tasks; the reflection model selects components and rewrites prompts.
target_config="qwen3.8-27b__semantic_layer"
reflection_model="gpt-6-sol"

# Put this on your large drive to store ALL run logs, artifacts and checkpoints.
results_dir="$project_root/benchmark-results"
# Example: results_dir="/Volumes/LargeDrive/SECHA-results"
progress_log=""  # Empty: keep the summary log inside the run directory.
split_file="$project_root/src/validation/optimization_splits/power-quality-monitoring_40-20-40.yaml"
components=()    # Empty: optimize all active prompts. Example: (query.instructions query_data.description)

max_metric_calls=100               # Candidate/case evaluations; not model requests.
reflection_minibatch_size=3
seed=0
model_request_max_attempts=20
model_request_retry_budget_seconds=1800
max_resumes=3                      # Automatic recovery after service failures.
resume_delay_seconds=60

usage() {
  cat <<USAGE
Usage: bash scripts/run_prompt_optimization.sh [OPTIONS]

Run the prompt optimizer. Edit the defaults at the top of this script, or pass:

  --config NAME                    Task model/capabilities (default: $target_config)
  --reflection-model NAME          Selection/rewrite model (default: $reflection_model)
  --reflection-provider NAME       Optional provider override for custom model names
  --results-dir PATH               All run files (default: $results_dir)
  --progress-log PATH              Optional separate summary log
  --split-file PATH                Train/validation/test YAML (default: 22/11/22 split)
  --components NAME [NAME ...]     Restrict editable prompts (default: all active)
  --max-metric-calls N              Search budget (default: $max_metric_calls)
  --reflection-minibatch-size N     Training cases per reflection (default: $reflection_minibatch_size)
  --seed N                         Sampling seed (default: $seed)
  --model-request-max-attempts N    HTTP retry ceiling (default: $model_request_max_attempts)
  --model-request-retry-budget-seconds N  Request deadline (default: $model_request_retry_budget_seconds)
  --max-resumes N                   Automatic restarts (default: $max_resumes)
  --resume-delay-seconds N          Restart delay (default: $resume_delay_seconds)
  --dry-run                        Validate/save inputs without model/database calls
  --resume RUN_DIR                 Continue a saved run using its frozen settings
  --list-components                Show editable component names and exit
  -h, --help                       Show this help

Examples:
  bash scripts/run_prompt_optimization.sh --results-dir "/Volumes/LargeDrive/SECHA-results" --dry-run
  bash scripts/run_prompt_optimization.sh --results-dir "/Volumes/LargeDrive/SECHA-results"
  bash scripts/run_prompt_optimization.sh --resume "/Volumes/LargeDrive/SECHA-results/power-quality-monitoring/gepa-RUN"

New runs use the target config's repetitions and concurrency (currently 3 and 4).
The default target has semantic-layer access and no fixed few-shot examples.
Resume ignores this script's experiment defaults; recovery/log flags still apply.
All optimizer flags are forwarded, including explicit --train/--validation/--test.
Relative paths are resolved from the project root. Requires uv.
USAGE
}

resume_requested=0
explicit_split=0
for argument in "$@"; do
  case "$argument" in
    -h|--help) usage; exit 0 ;;
    --resume|--resume=*) resume_requested=1 ;;
    --train|--train=*|--validation|--validation=*|--test|--test=*) explicit_split=1 ;;
  esac
done

cd -- "$project_root"
command=(uv run --locked --no-dev --no-cache --group optimization python -u
  -m src.validation.run_optimization
  --max-resumes "$max_resumes"
  --resume-delay-seconds "$resume_delay_seconds")

if [[ "$resume_requested" -eq 0 ]]; then
  command+=(
    --config "$target_config"
    --reflection-model "$reflection_model"
    --results-dir "$results_dir"
    --max-metric-calls "$max_metric_calls"
    --reflection-minibatch-size "$reflection_minibatch_size"
    --seed "$seed"
    --model-request-max-attempts "$model_request_max_attempts"
    --model-request-retry-budget-seconds "$model_request_retry_budget_seconds")
  if [[ "$explicit_split" -eq 0 ]]; then
    command+=(--split-file "$split_file")
  fi
  if [[ ${#components[@]} -gt 0 ]]; then
    command+=(--components "${components[@]}")
  fi
fi
if [[ -n "$progress_log" ]]; then
  command+=(--progress-log "$progress_log")
fi

# Scalar flags supplied by the user take precedence over the defaults above.
# Arrays preserve spaces in paths; exec preserves exit status and Ctrl-C behavior.
command+=("$@")
printf 'Starting prompt optimizer from %s\n' "$project_root"
exec "${command[@]}"
