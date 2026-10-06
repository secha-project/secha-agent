#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_benchmarks.sh [--new | --resume DIR] [--dry-run] [--skip CONFIG]...

Run all YAML configs in order, preserving each config's tasks and repetitions.
The first invocation starts from scratch. Repeat the same command after an
interruption: finished configs are skipped, the interrupted config resumes its
unfinished attempts, then the remaining configs run. Old standalone results
are never used to infer completion.

  bash scripts/run_benchmarks.sh --dry-run     # Validate configs without calls
  bash scripts/run_benchmarks.sh              # Start or continue the saved queue
  bash scripts/run_benchmarks.sh --new        # Start every config from scratch
  bash scripts/run_benchmarks.sh --resume benchmark-results/batch-logs/SAVED_DIR

--skip CONFIG omits a config when creating a queue (repeatable; .yaml optional).
Use --new to change an existing queue. Relative paths resolve from the project
root. --dry-run always validates a new selection and writes no state or logs.

Live progress remains on the terminal. Run directories, completion state, YAML
snapshots and compact logs are saved under benchmark-results/batch-logs/.
A failure or Ctrl-C stops the queue; rerun this script to continue. A completed
queue stays complete until --new is requested. --resume selects an older queue.
USAGE
}

dry_run=0
new_queue=0
logs_dir=""
skip_configs=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --new) new_queue=1; shift ;;
    --resume|--skip)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "$1 requires a value." >&2
        exit 2
      fi
      if [[ "$1" == --resume ]]; then
        logs_dir=$2
      else
        skip_config=${2%.yaml}
        skip_configs+=("${skip_config%.yml}")
      fi
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -n "$logs_dir" && ( "$new_queue" -eq 1 || "$dry_run" -eq 1 ) ]]; then
  echo "--resume cannot be combined with --new or --dry-run." >&2
  exit 2
fi

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

latest_file=benchmark-results/batch-logs/latest-queue
if [[ "$dry_run" -eq 0 && "$new_queue" -eq 0 && -z "$logs_dir" && -f "$latest_file" ]]; then
  IFS= read -r logs_dir < "$latest_file"
  if [[ -z "$logs_dir" ]]; then
    echo "Empty saved queue path in $latest_file. Use --new to start over." >&2
    exit 2
  fi
fi

pending_configs=()
if [[ -n "$logs_dir" ]]; then
  if [[ ! -s "$logs_dir/queue.txt" ]]; then
    echo "No saved queue at $logs_dir. Use --new to start from scratch." >&2
    exit 2
  fi
  while IFS= read -r config_name; do
    if [[ -z "$config_name" || "$config_name" == */* ]]; then
      echo "Invalid config in saved queue: $config_name" >&2
      exit 2
    fi
    for skip_config in "${skip_configs[@]-}"; do
      if [[ "$config_name" == "$skip_config" ]]; then
        echo "Cannot change a saved queue with --skip. Use --new --skip $skip_config." >&2
        exit 2
      fi
    done
    if ! cmp -s -- "src/validation/run_configs/$config_name.yaml" "$logs_dir/configs/$config_name.yaml"; then
      echo "Config changed or missing: $config_name. Restore it to resume, or use --new." >&2
      exit 2
    fi
    pending_configs+=("$config_name")
  done < "$logs_dir/queue.txt"
  printf '\nContinuing saved queue: %s\n' "$logs_dir"
else
  for config in "${configs[@]}"; do
    config_name=${config##*/}
    config_name=${config_name%.yaml}
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
fi

total=${#pending_configs[@]}
if [[ "$total" -eq 0 ]]; then
  echo "No configurations remain after the requested skips."
  exit 0
fi
printf '\nSaved order (%d configurations):\n' "$total"
printf '  %s\n' "${pending_configs[@]}"
if [[ "$dry_run" -eq 0 ]]; then
  if [[ -z "$logs_dir" ]]; then
    logs_dir="benchmark-results/batch-logs/$(date -u +%Y-%m-%d_%H-%M-%SZ)_$$"
    mkdir -p -- "$logs_dir/configs"
    for config_name in "${pending_configs[@]}"; do
      cp -- "src/validation/run_configs/$config_name.yaml" "$logs_dir/configs/"
    done
    printf '%s\n' "${pending_configs[@]}" > "$logs_dir/queue.txt"
  fi
  logs_dir=$(cd -- "$logs_dir" && pwd)
  # Publish only a fully initialized queue; interruption never exposes half a file.
  mkdir -p -- "${latest_file%/*}"
  printf '%s\n' "$logs_dir" > "$latest_file.tmp.$$"
  mv -- "$latest_file.tmp.$$" "$latest_file"
  printf '\nState and logs: %s\n' "$logs_dir"
fi

batch_started=$SECONDS
position=0
current_config=""
on_interrupt() {
  printf '\nInterrupted while running %s. Rerun this script to continue.\n' "$current_config" >&2
  if [[ -n "${log_file:-}" && "$log_file" != /dev/null ]]; then
    printf '[%d/%d] INTERRUPTED: %s\n' "$position" "$total" "$current_config" >> "$log_file"
  fi
  exit "$1"
}
trap 'on_interrupt 130' INT
trap 'on_interrupt 143' TERM

for current_config in "${pending_configs[@]}"; do
  position=$((position + 1))
  if [[ "$dry_run" -eq 0 && -f "$logs_dir/$current_config.done" ]]; then
    printf '[%d/%d] SKIP completed: %s\n' "$position" "$total" "$current_config"
    continue
  fi
  started=$SECONDS
  log_file=/dev/null
  action=RUNNING
  command=(uv run --locked --no-dev --no-cache python -u
    -m src.validation.run_benchmark --config "$current_config")
  if [[ "$dry_run" -eq 1 ]]; then
    action=VALIDATING
    command+=(--dry-run)
  else
    printf -v log_file '%s/%02d_%s.log' "$logs_dir" "$position" "$current_config"
    if [[ -f "$logs_dir/$current_config.json" ]]; then
      action=RESUMING
    fi
    command+=(--progress-log "$log_file" --run-state "$logs_dir/$current_config.json")
    printf '[%d/%d] %s: %s\n' "$position" "$total" "$action" "$current_config" >> "$log_file"
  fi

  printf '\n======================================================================\n'
  printf '[%d/%d] %s: %s\n' "$position" "$total" "$action" "$current_config"
  printf 'Started: %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  if [[ "$dry_run" -eq 0 ]]; then
    printf 'Log: %s\n' "$log_file"
  fi
  printf '======================================================================\n\n'

  # Keep stdout/stderr attached for live progress; Python handles the checkpoint
  # and locks each saved run so concurrent launchers cannot repeat paid work.
  if "${command[@]}"; then
    elapsed=$((SECONDS - started))
    printf '\n[%d/%d] DONE: %s (%dm %02ds)\n' \
      "$position" "$total" "$current_config" "$((elapsed / 60))" "$((elapsed % 60))"
    if [[ "$dry_run" -eq 0 ]]; then
      touch -- "$logs_dir/$current_config.done"
      printf '[%d/%d] DONE: %s (%ds)\n' \
        "$position" "$total" "$current_config" "$elapsed" >> "$log_file"
    fi
  else
    status=$?
    printf '\n[%d/%d] FAILED: %s (exit %d). Queue stopped.\n' \
      "$position" "$total" "$current_config" "$status" >&2
    if [[ "$dry_run" -eq 0 ]]; then
      printf '[%d/%d] STOPPED: %s (exit %d)\n' \
        "$position" "$total" "$current_config" "$status" >> "$log_file"
      printf 'Inspect log: %s\nRerun: bash scripts/run_benchmarks.sh\n' "$log_file" >&2
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
  printf 'Logs: %s\nUse --new to start all selected configs from scratch.\n' "$logs_dir"
fi
