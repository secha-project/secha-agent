"""Generate a pass@1 LaTeX tabular for one ablation from completed runs.

Usage: python -m scripts.generate_pass_rate_table -m 20
       python -m scripts.generate_pass_rate_table --ablation few_shot -m 15
       python -m scripts.generate_pass_rate_table --best-ablation -m 15
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from textwrap import wrap

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from scripts.bootstrap_pass_rate import TaskScore, bootstrap_pass_rate, load_task_scores
from scripts.load_benchmark_runs import (
    DEFAULT_RESULTS_DIR,
    BenchmarkRun,
    load_benchmark_runs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / "docs/results/templates"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/results/generated/pass_rates.tex"
BEST_MODEL_ORDER = ("qwen3.8-27b", "gpt-6-luna", "glm-5.3", "gpt-6-sol")
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


def _rate(value: float) -> str:
    return f"{value:.3f}"


def _category_header(category: str) -> str:
    words = category.title().split(" ", 1)
    if len(words) == 1:
        return r"\textbf{" + latex_escape(words[0]) + "}"
    return (
        r"\textbf{" + latex_escape(words[0]) + r"}\\[-3pt]\textbf{"
        + latex_escape(words[1]) + "}"
    )


def _category_rates(scores: Sequence[TaskScore]) -> dict[str, float]:
    scores_by_category: dict[str, list[float]] = defaultdict(list)
    for score in scores:
        scores_by_category[score.category].append(score.pass_at_one)
    return {
        category: sum(values) / len(values)
        for category, values in scores_by_category.items()
    }


def select_best_ablation_runs(runs: Sequence[BenchmarkRun]) -> list[BenchmarkRun]:
    """Select each root's highest unrounded category-balanced pass@1.

    All candidates must have comparable tasks, datasets and repetitions.
    Exact ties use the alphabetically first run-config name.
    """
    candidates = [
        run for run in runs
        if run.name.rpartition("__")[0] and run.name.rpartition("__")[2]
    ]
    if not candidates:
        raise ValueError("No completed ROOT__ABLATION runs were found")
    if len({run.name for run in candidates}) != len(candidates):
        raise ValueError("Ablation candidates must have distinct run_config names")
    if len({(run.dataset, run.repetitions) for run in candidates}) != 1:
        raise ValueError("Ablation candidates must use the same dataset and repetitions")

    identities: dict[str, tuple[str, str]] = {}
    best: dict[str, tuple[float, BenchmarkRun]] = {}
    task_categories: dict[str, str] | None = None
    for run in sorted(candidates, key=lambda item: item.name):
        root = run.name.rpartition("__")[0]
        identity = (run.model_provider, run.model_name)
        if identities.setdefault(root, identity) != identity:
            raise ValueError(f"Ablations under {root!r} must use the same model")
        scores = load_task_scores(run.path)
        current_tasks = {score.case_id: score.category for score in scores}
        if task_categories is None:
            task_categories = current_tasks
        elif current_tasks != task_categories:
            raise ValueError("Ablation candidates must contain the same tasks and categories")
        rates = _category_rates(scores)
        estimate = sum(rates.values()) / len(rates)
        if root not in best or estimate > best[root][0]:
            best[root] = (estimate, run)
    return [best[root][1] for root in sorted(best)]


def render_pass_rate_table(
    runs: Sequence[BenchmarkRun],
    *,
    tasks_per_category: int,
    ablation: str | None = None,
    best_ablation: bool = False,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> str:
    """Render Table 1 as a tabular fragment, using equal category weights."""
    if best_ablation and ablation is not None:
        raise ValueError("Select one ablation or the best ablation per model")
    if best_ablation:
        selected = select_best_ablation_runs(runs)
    elif ablation is None:
        selected = [
            run for run in runs if run.enabled_capabilities == BASELINE_CAPABILITIES
        ]
    else:
        if not ablation or not all(char.isalnum() or char in "_-" for char in ablation):
            raise ValueError("Ablation must contain only letters, digits, '_' or '-'")
        selected = [
            run for run in runs
            if run.name.rpartition("__")[0] and run.name.rpartition("__")[2] == ablation
        ]
    if not selected:
        raise ValueError(f"No completed runs for ablation {ablation or 'baseline'!r}")
    if len({run.name for run in selected}) != len(selected):
        raise ValueError("Compared runs must have distinct run_config names")
    if len({run.dataset for run in selected}) != 1:
        raise ValueError("Compared runs must use the same dataset")

    if best_ablation:
        model_rank = {name: index for index, name in enumerate(BEST_MODEL_ORDER)}
        selected = sorted(
            selected,
            key=lambda run: (
                model_rank.get(run.name.rpartition("__")[0], len(BEST_MODEL_ORDER)),
                run.name,
            ),
        )
    else:
        selected = sorted(selected, key=lambda run: run.name)
    category_names: set[str] | None = None
    task_categories: dict[str, str] | None = None
    rows: list[dict[str, object]] = []
    for run in selected:
        scores = load_task_scores(run.path)
        current_categories = {score.category for score in scores}
        current_tasks = {score.case_id: score.category for score in scores}
        if category_names is None:
            category_names = current_categories
            task_categories = current_tasks
        elif current_categories != category_names or current_tasks != task_categories:
            raise ValueError("Compared runs must contain the same tasks and categories")

        category_rates = _category_rates(scores)
        overall = bootstrap_pass_rate(
            scores,
            tasks_per_category=tasks_per_category,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        )
        rows.append(
            {
                "name": run.name.rpartition("__")[0] or run.name,
                "ablation_lines": wrap(
                    run.name.rpartition("__")[2].replace("_", " ").replace("-", " "),
                    width=15,
                ),
                "category_rates": category_rates,
                "overall": _rate(overall.estimate),
                "lower": _rate(overall.confidence_interval[0]),
                "upper": _rate(overall.confidence_interval[1]),
            }
        )

    assert category_names is not None
    categories = [category for category in CATEGORY_ORDER if category in category_names]
    categories.extend(sorted(category_names - set(categories)))
    for row in rows:
        rates = row["category_rates"]
        assert isinstance(rates, dict)
        row["category_rates"] = [_rate(rates[category]) for category in categories]

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
        include_ablation=best_ablation,
        rows=rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ablation", help="Run-config suffix after '__'; defaults to baseline capabilities")
    selection.add_argument("--best-ablation", action="store_true", help="Select the highest overall pass@1 ablation per model")
    parser.add_argument("--output", type=Path, help="LaTeX path; defaults to pass_rates[_<ablation>|_best].tex")
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
        ablation=args.ablation,
        best_ablation=args.best_ablation,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    suffix = "_best" if args.best_ablation else (
        "" if args.ablation in (None, "baseline") else f"_{args.ablation}"
    )
    output = args.output or DEFAULT_OUTPUT.with_name(f"pass_rates{suffix}.tex")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
