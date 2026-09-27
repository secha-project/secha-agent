"""Generate Table 3: paired pass@1 differences for one ablation.

Usage: python -m scripts.generate_pairwise_pass_rate_table --ablation baseline -m 20
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from scripts.bootstrap_pass_rate import (
    bootstrap_pass_rate_difference,
    load_task_scores,
)
from scripts.generate_pass_rate_table import latex_escape
from scripts.load_benchmark_runs import (
    DEFAULT_RESULTS_DIR,
    BenchmarkRun,
    load_benchmark_runs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / "docs/results/templates"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/results/generated/pairwise_pass_rates.tex"


def _selected_runs(runs: Sequence[BenchmarkRun], ablation: str) -> list[BenchmarkRun]:
    if not ablation or not all(char.isalnum() or char in "_-" for char in ablation):
        raise ValueError("Ablation must contain only letters, digits, '_' or '-'")
    by_root: dict[str, BenchmarkRun] = {}
    for run in runs:
        root, separator, suffix = run.name.rpartition("__")
        if not separator or not root or suffix != ablation:
            continue
        if root in by_root:
            raise ValueError(f"Multiple completed runs for {root}__{ablation}")
        by_root[root] = run
    if len(by_root) < 2:
        raise ValueError(f"At least two completed runs are needed for {ablation!r}")
    selected = sorted(by_root.values(), key=lambda run: (run.model_name, run.name))
    if len({run.dataset for run in selected}) != 1:
        raise ValueError("Compared runs must use the same dataset")
    if len({run.repetitions for run in selected}) != 1:
        raise ValueError("Compared runs must use the same number of repetitions")
    if len({run.model_name for run in selected}) != len(selected):
        raise ValueError("Compared runs must have distinct model names")
    return selected


def _percentage_points(value: float) -> str:
    rounded = round(100 * value, 1)
    return "0.0" if rounded == 0 else f"{rounded:+.1f}"


def render_pairwise_pass_rate_table(
    runs: Sequence[BenchmarkRun],
    *,
    ablation: str,
    tasks_per_category: int,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> str:
    """Render all model pairs as Model 1 minus Model 2, in percentage points."""
    selected = _selected_runs(runs, ablation)
    scores_by_name = {run.name: load_task_scores(run.path) for run in selected}
    rows: list[dict[str, str]] = []
    for first, second in combinations(selected, 2):
        result = bootstrap_pass_rate_difference(
            scores_by_name[first.name],
            scores_by_name[second.name],
            tasks_per_category=tasks_per_category,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        )
        lower, upper = result.confidence_interval
        rows.append(
            {
                "first": first.model_name,
                "second": second.model_name,
                "difference": _percentage_points(result.estimate),
                "lower": _percentage_points(lower),
                "upper": _percentage_points(upper),
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
    return environment.get_template("pairwise_pass_rates.tex.j2").render(
        confidence_header=rf"\textbf{{{100 * confidence_level:g}\% CI}}", rows=rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ablation", required=True, help="Run-config suffix after '__'"
    )
    parser.add_argument(
        "-m",
        "--tasks-per-category",
        type=int,
        required=True,
        help="Number of tasks drawn per category in each bootstrap sample.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    table = render_pairwise_pass_rate_table(
        load_benchmark_runs(args.results_dir),
        ablation=args.ablation,
        tasks_per_category=args.tasks_per_category,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
