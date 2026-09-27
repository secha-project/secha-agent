"""Generate the baseline pass@1 LaTeX tabular from completed benchmark runs.

Usage: python -m scripts.generate_pass_rate_table -m 20
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from scripts.bootstrap_pass_rate import bootstrap_pass_rate, load_task_scores
from scripts.load_benchmark_runs import (
    DEFAULT_RESULTS_DIR,
    BenchmarkRun,
    load_benchmark_runs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / "docs/results/templates"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/results/generated/pass_rates.tex"
BASELINE_CAPABILITIES = frozenset(
    {"catalog", "query", "table_outputs", "visualization"}
)
CATEGORY_ORDER = (
    "data discovery",
    "data transformations",
    "semantic integration",
    "domain knowledge",
)
_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(value: str) -> str:
    return "".join(_LATEX_ESCAPES.get(character, character) for character in value)


def _percent(value: float) -> str:
    return f"{100 * value:.1f}"


def _category_header(category: str) -> str:
    words = category.title().split(" ", 1)
    if len(words) == 1:
        return r"\textbf{" + latex_escape(words[0]) + "}"
    return (
        r"\textbf{" + latex_escape(words[0]) + r"}\\[-3pt]\textbf{"
        + latex_escape(words[1]) + "}"
    )


def render_pass_rate_table(
    runs: Sequence[BenchmarkRun],
    *,
    tasks_per_category: int,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> str:
    """Render Table 1 as a tabular fragment, using equal category weights."""
    baseline = [
        run for run in runs if run.enabled_capabilities == BASELINE_CAPABILITIES
    ]
    if not baseline:
        raise ValueError("No completed baseline runs were found")
    if len({run.name for run in baseline}) != len(baseline):
        raise ValueError("Baseline runs must have distinct run_config names")
    if len({run.dataset for run in baseline}) != 1:
        raise ValueError("Baseline runs must use the same dataset")

    baseline = sorted(baseline, key=lambda run: run.name)
    category_names: set[str] | None = None
    task_categories: dict[str, str] | None = None
    rows: list[dict[str, object]] = []
    for run in baseline:
        scores = load_task_scores(run.path)
        current_categories = {score.category for score in scores}
        current_tasks = {score.case_id: score.category for score in scores}
        if category_names is None:
            category_names = current_categories
            task_categories = current_tasks
        elif current_categories != category_names or current_tasks != task_categories:
            raise ValueError("Baseline runs must contain the same tasks and categories")

        scores_by_category: dict[str, list[float]] = defaultdict(list)
        for score in scores:
            scores_by_category[score.category].append(score.pass_at_one)
        category_rates = {
            category: sum(values) / len(values)
            for category, values in scores_by_category.items()
        }
        overall = bootstrap_pass_rate(
            scores,
            tasks_per_category=tasks_per_category,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        )
        rows.append(
            {
                "name": run.name,
                "category_rates": category_rates,
                "overall": _percent(overall.estimate),
                "lower": _percent(overall.confidence_interval[0]),
                "upper": _percent(overall.confidence_interval[1]),
            }
        )

    assert category_names is not None
    categories = [category for category in CATEGORY_ORDER if category in category_names]
    categories.extend(sorted(category_names - set(categories)))
    for row in rows:
        rates = row["category_rates"]
        assert isinstance(rates, dict)
        row["category_rates"] = [_percent(rates[category]) for category in categories]

    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=False,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters["latex_escape"] = latex_escape
    return environment.get_template("pass_rates.tex.j2").render(
        category_headers=[_category_header(category) for category in categories],
        confidence_header=rf"\textbf{{{100 * confidence_level:g}\% CI}}",
        rows=rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "-m", "--tasks-per-category", type=int, required=True,
        help="Number of tasks drawn per category in each bootstrap sample.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    table = render_pass_rate_table(
        load_benchmark_runs(args.results_dir),
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
