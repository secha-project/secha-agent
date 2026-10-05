#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_benchmarks.sh [--dry-run] [--skip CONFIG]... [--redo-qwen-tables]

Run YAML configurations sequentially, preserving their configured repetitions.
Repeat --skip to omit configurations you have already completed. CONFIG may be
a config stem or filename, for example glm-5.3__baseline or its .yaml filename.

  bash scripts/run_benchmarks.sh --skip glm-5.3__baseline
  bash scripts/run_benchmarks.sh --skip glm-5.3__baseline --dry-run
  bash scripts/run_benchmarks.sh --redo-qwen-tables --dry-run
  bash scripts/run_benchmarks.sh --redo-qwen-tables

--redo-qwen-tables reruns all table-output tasks in the four original Qwen
ablations. Each redo copies its original checkpoint and artifacts into a new
run directory, retains other attempts, and generates a full report. The current
dataset selects 25 tasks x 3 repetitions x 4 ablations = 300 new attempts.

Each run prints a numbered banner and preserves its live progress display.
Live runs save compact status logs under benchmark-results/batch-logs/, with
progress once per minute. The batch stops on the first failed command.
--dry-run validates only; it creates no run logs.
USAGE
}

dry_run=0
redo_qwen_tables=0
skip_configs=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --redo-qwen-tables) redo_qwen_tables=1; shift ;;
    --skip)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "--skip requires a config name." >&2
        exit 2
      fi
      skip_config=${2%.yaml}
      skip_configs+=("${skip_config%.yml}")
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
shopt -s nullglob
configs=(src/validation/run_configs/*.yaml)
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No benchmark configs found." >&2
  exit 1
fi

# Reject typos before starting any paid benchmark work.
for skip_config in "${skip_configs[@]-}"; do
  [[ -n "$skip_config" ]] || continue
  if [[ "$skip_config" == */* || ! -f "src/validation/run_configs/$skip_config.yaml" ]]; then
    echo "Unknown config in --skip: $skip_config" >&2
    exit 2
  fi
done

pending_configs=()
for config in "${configs[@]}"; do
  config_name=${config##*/}
  config_name=${config_name%.yaml}
  if [[ "$redo_qwen_tables" -eq 1 && "$config_name" != qwen3.8-27b__* ]]; then
    continue
  fi
  skip=0
  for skip_config in "${skip_configs[@]-}"; do
    if [[ "$config_name" == "$skip_config" ]]; then
      skip=1
      break
    fi
  done
  if [[ "$skip" -eq 1 ]]; then
    printf 'SKIP  %s\n' "$config_name"
  else
    pending_configs+=("$config_name")
  fi
done

total=${#pending_configs[@]}
if [[ "$total" -eq 0 ]]; then
  echo "No configurations remain after the requested skips."
  exit 0
fi

redo_source() {
  # Pin the original runs rather than choosing whichever redo is newest.
  case "$1" in
    qwen3.8-27b__baseline) echo "benchmark-results/power-quality-monitoring/qwen3.8-27b__baseline_2026-09-28_03-23" ;;
    qwen3.8-27b__few_shot) echo "benchmark-results/power-quality-monitoring/qwen3.8-27b__few_shot_2026-09-28_17-43" ;;
    qwen3.8-27b__few_shot_semantic_layer) echo "benchmark-results/power-quality-monitoring/qwen3.8-27b__few_shot_semantic_layer_2026-09-28_10-35" ;;
    qwen3.8-27b__semantic_layer) echo "benchmark-results/power-quality-monitoring/qwen3.8-27b__semantic_layer_2026-09-29_08-28" ;;
    *) echo "No original redo source configured for $1." >&2; return 2 ;;
  esac
}

if [[ "$redo_qwen_tables" -eq 1 ]]; then
  for config_name in "${pending_configs[@]}"; do
    source_dir=$(redo_source "$config_name")
    if [[ ! -f "$source_dir/checkpoint.json" || ! -d "$source_dir/artifacts" ]]; then
      echo "Missing original checkpoint or artifacts: $source_dir" >&2
      exit 2
    fi
  done
  printf '\nRedoing table tasks only; other attempts are retained.\n'
fi

printf '\nQueued %d of %d configurations, in this order:\n' "$total" "${#configs[@]}"
printf '  %s\n' "${pending_configs[@]}"
logs_dir=""
if [[ "$dry_run" -eq 0 ]]; then
  logs_dir="benchmark-results/batch-logs/$(date -u +%Y-%m-%d_%H-%M-%SZ)_$$"
  mkdir -p -- "$logs_dir"
  printf '\nBatch logs: %s/%s\n' "$PWD" "$logs_dir"
fi

batch_started=$SECONDS
position=0
current_config=""
on_interrupt() {
  printf '\nInterrupted while running %s. Batch stopped.\n' "$current_config" >&2
  if [[ -n "${log_file:-}" && "$log_file" != /dev/null ]]; then
    printf '[%d/%d] INTERRUPTED: %s\n' "$position" "$total" "$current_config" >> "$log_file"
  fi
  exit 130
}
trap on_interrupt INT

for current_config in "${pending_configs[@]}"; do
  position=$((position + 1))
  started=$SECONDS
  log_file=/dev/null
  action=RUNNING
  command=(uv run --locked --no-dev --no-cache python -u
    -m src.validation.run_benchmark --config "$current_config")
  if [[ "$redo_qwen_tables" -eq 1 ]]; then
    command+=(--redo-tables)
    source_dir=$(redo_source "$current_config")
    printf '\nRedo source: %s/%s\n' "$PWD" "$source_dir"
    if [[ "$dry_run" -eq 0 ]]; then
      command+=(--resume "$source_dir")
    fi
  fi
  if [[ "$dry_run" -eq 1 ]]; then
    action=VALIDATING
    command+=(--dry-run)
  else
    printf -v log_file '%s/%02d_%s.log' "$logs_dir" "$position" "$current_config"
    command+=(--progress-log "$log_file")
    printf '[%d/%d] RUNNING: %s\n' "$position" "$total" "$current_config" >> "$log_file"
  fi

  printf '\n======================================================================\n'
  printf '[%d/%d] %s: %s\n' "$position" "$total" "$action" "$current_config"
  printf 'Started: %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  if [[ "$dry_run" -eq 0 ]]; then
    printf 'Log: %s/%s\n' "$PWD" "$log_file"
  fi
  printf '======================================================================\n\n'

  # Keep stdout/stderr attached to the terminal for live progress. The runner
  # writes periodic checkpoint counts separately, without recording redraws.
  if "${command[@]}"; then
    elapsed=$((SECONDS - started))
    printf '\n[%d/%d] DONE: %s (%dm %02ds)\n' \
      "$position" "$total" "$current_config" "$((elapsed / 60))" "$((elapsed % 60))"
    if [[ "$dry_run" -eq 0 ]]; then
      printf '[%d/%d] DONE: %s (%ds)\n' \
        "$position" "$total" "$current_config" "$elapsed" >> "$log_file"
    fi
  else
    status=$?
    printf '\n[%d/%d] FAILED: %s (exit %d). Batch stopped.\n' \
      "$position" "$total" "$current_config" "$status" >&2
    if [[ "$dry_run" -eq 0 ]]; then
      printf '[%d/%d] STOPPED: %s (exit %d)\n' \
        "$position" "$total" "$current_config" "$status" >> "$log_file"
      printf 'Inspect log: %s/%s\n' "$PWD" "$log_file" >&2
    fi
    exit "$status"
  fi
done

elapsed=$((SECONDS - batch_started))
printf '\nFinished %d configurations in %dh %02dm %02ds.\n' \
  "$total" "$((elapsed / 3600))" "$((elapsed / 60 % 60))" "$((elapsed % 60))"
if [[ "$dry_run" -eq 1 ]]; then
  echo "Dry run only: no model/database calls or benchmark results."
else
  printf 'Logs: %s/%s\n' "$PWD" "$logs_dir"
fi
