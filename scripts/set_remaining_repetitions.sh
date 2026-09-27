#!/usr/bin/env bash
# Run from the project root on the benchmark VM.
set -euo pipefail

config_dir=src/validation/run_configs

# uv and its Python child may both appear; deduplicate their config names.
running=$(ps -eo args= | awk '
  {
    for (i = 1; i < NF; i++) {
      if ($i == "-m" && $(i + 1) == "src.validation.run_benchmark") {
        for (j = i + 2; j < NF; j++) {
          if ($j == "--config") print $(j + 1)
        }
      }
    }
  }
' | sort -u)
running=${running%.yaml}
running=${running%.yml}

if [[ -z "$running" || ! -f "$config_dir/$running.yaml" ]]; then
  echo "Could not identify exactly one running config. No files changed." >&2
  exit 1
fi

echo "Keeping $running.yaml unchanged."
for config in "$config_dir"/*.yaml; do
  [[ "$config" == "$config_dir/$running.yaml" ]] && continue
  # Preserve the original backup if the script is run again.
  [[ -e "$config.bak" ]] || cp -p -- "$config" "$config.bak"
  sed -E 's/^repetitions:[[:space:]]*[0-9]+[[:space:]]*$/repetitions: 3/' \
    "$config" > "$config.tmp"
  mv -- "$config.tmp" "$config"
  echo "Set ${config##*/} to 3 repetitions."
done
