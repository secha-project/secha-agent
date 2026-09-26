#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--dry-run" ) ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
shopt -s nullglob
configs=(src/validation/run_configs/*.yaml)
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No benchmark configs found." >&2
  exit 1
fi

for config in "${configs[@]}"; do
  echo "Running ${config##*/}"
  uv run --locked --no-dev --no-cache python \
    -m src.validation.run_benchmark --config "${config##*/}" "$@"
done
