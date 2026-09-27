"""Generate Table 2: tool calls and latency per task attempt for one ablation.

Usage: python -m scripts.generate_execution_stats_table --ablation baseline
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from scripts.generate_pass_rate_table import latex_escape
from scripts.load_benchmark_runs import (
    DEFAULT_RESULTS_DIR,
    BenchmarkRun,
    load_benchmark_runs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / "docs/results/templates"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/results/generated/execution_stats.tex"


def _nonnegative_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be a finite nonnegative number")
    return float(value)


def _attempts(run: BenchmarkRun) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for category in run.results["categories"].values():
        for task in category["tasks"].values():
            attempts.extend(task["attempts"])
    return attempts


def render_execution_stats_table(runs: Sequence[BenchmarkRun], *, ablation: str) -> str:
    """Render mean tool calls and mean/p95 latency for one ablation."""
    if not ablation or not all(char.isalnum() or char in "_-" for char in ablation):
        raise ValueError("Ablation must contain only letters, digits, '_' or '-'")
    selected: dict[str, BenchmarkRun] = {}
    for run in runs:
        root, separator, suffix = run.name.rpartition("__")
        if not separator or not root or suffix != ablation:
            continue
        if root in selected:
            raise ValueError(f"Multiple completed runs for {root}__{ablation}")
        selected[root] = run
    if not selected:
        raise ValueError(f"No completed runs for ablation {ablation!r}")
    if len({run.dataset for run in selected.values()}) != 1:
        raise ValueError("Compared runs must use the same dataset")
    if len({run.repetitions for run in selected.values()}) != 1:
        raise ValueError("Compared runs must use the same number of repetitions")

    rows: list[dict[str, object]] = []
    for run in sorted(selected.values(), key=lambda item: (item.model_name, item.name)):
        attempts = _attempts(run)
        if not attempts:
            raise ValueError(f"Run {run.name!r} has no attempts")
        durations: list[float] = []
        tool_calls = 0.0
        for index, attempt in enumerate(attempts, start=1):
            label = f"Run {run.name!r} attempt {index}"
            durations.append(
                _nonnegative_number(
                    attempt.get("task_duration"), f"{label} task_duration"
                )
            )
            metrics = attempt.get("metrics")
            if not isinstance(metrics, dict):
                raise TypeError(f"{label} metrics must be an object")
            tool_calls += _nonnegative_number(
                metrics.get("tool_calls", 0), f"{label} tool_calls"
            )

        rows.append(
            {
                "model": run.model_name,
                "tool_calls": f"{tool_calls / len(attempts):.1f}",
                "duration": f"{sum(durations) / len(durations):.1f}",
                # Nearest rank: at least 95% of observed attempts meet this latency.
                "duration_p95": f"{sorted(durations)[math.ceil(0.95 * len(durations)) - 1]:.1f}",
            }
        )

    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=False,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters["latex_escape"] = latex_escape
    return environment.get_template("execution_stats.tex.j2").render(rows=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ablation", required=True, help="Run-config suffix after '__'"
    )
    args = parser.parse_args()

    table = render_execution_stats_table(
        load_benchmark_runs(args.results_dir), ablation=args.ablation
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
