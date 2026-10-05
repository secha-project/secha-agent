"""Plot Figure 3: average pass@1 versus estimated total model cost.

Run configs must be named ROOT__ABLATION, for example model-a__baseline.

Usage: python -m scripts.generate_pass_rate_cost_figure --ablation baseline -m 20
       python -m scripts.generate_pass_rate_cost_figure --best-ablation -m 15
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    coord_cartesian,
    element_blank,
    element_line,
    element_rect,
    element_text,
    geom_errorbar,
    geom_point,
    geom_text,
    ggplot,
    labs,
    scale_color_manual,
    scale_shape_manual,
    scale_y_continuous,
    theme,
    theme_minimal,
)

from scripts.bootstrap_pass_rate import bootstrap_pass_rate, load_task_scores
from scripts.generate_ablation_figure import split_ablation_name
from scripts.generate_pass_rate_table import (
    BEST_MODEL_ORDER,
    select_best_ablation_runs,
)
from scripts.load_benchmark_runs import (
    DEFAULT_RESULTS_DIR,
    BenchmarkRun,
    _load_run,
    load_benchmark_runs,
)
from scripts.report_figure_style import (
    DEFAULT_TYPOGRAPHY,
    ReportTypography,
    add_figure_arguments,
    copy_figure_to_thesis,
    figure_dimensions,
    typography_from_args,
    typography_theme,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/pdf"
COLORS = (
    "#277da1",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#9c6ade",
    "#8c564b",
    "#17a2a4",
    "#ba6e93",
)
MARKERS = ("o", "s", "^", "D", "P", "v", "X", "*")


def select_ablation_runs(
    runs: Sequence[BenchmarkRun], ablation: str
) -> dict[str, BenchmarkRun]:
    """Select one completed run per root for an ablation suffix."""
    if not ablation or not all(char.isalnum() or char in "_-" for char in ablation):
        raise ValueError("Ablation must contain only letters, digits, '_' or '-'")
    selected: dict[str, BenchmarkRun] = {}
    for run in runs:
        parsed = split_ablation_name(run.name)
        if parsed is None or parsed[1] != ablation:
            continue
        root = parsed[0]
        if root in selected:
            raise ValueError(f"Multiple completed runs for {root}__{ablation}")
        selected[root] = run
    if not selected:
        raise ValueError(f"No completed runs for ablation {ablation!r}")
    if len({run.dataset for run in selected.values()}) != 1:
        raise ValueError("Compared runs must use the same dataset")
    if len({run.repetitions for run in selected.values()}) != 1:
        raise ValueError("Compared runs must use the same number of repetitions")
    return dict(sorted(selected.items()))


def _total_cost_usd(run: BenchmarkRun) -> float:
    usage = run.results["overall"].get("usage")
    cost = usage.get("estimated_cost_usd") if isinstance(usage, dict) else None
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(cost)
        or cost < 0
    ):
        raise ValueError(f"Run {run.name!r} needs a finite estimated_cost_usd")
    return float(cost)


def replace_run_reports(
    runs: Sequence[BenchmarkRun], paths: Sequence[Path]
) -> list[BenchmarkRun]:
    """Use explicitly selected report versions without counting a run twice."""
    overrides: dict[tuple[str, str], BenchmarkRun] = {}
    available = {(run.dataset, run.name) for run in runs}
    for path in paths:
        run = _load_run(path)
        if run is None:
            raise ValueError(f"Invalid or incomplete report override: {path}")
        key = (run.dataset, run.name)
        if key not in available:
            raise ValueError(f"Report override has no matching loaded run: {path}")
        if key in overrides:
            raise ValueError(f"Multiple report overrides for {run.name!r}")
        overrides[key] = run
    return [overrides.get((run.dataset, run.name), run) for run in runs]


def pass_rate_cost_data(
    runs: dict[str, BenchmarkRun],
    *,
    tasks_per_category: int,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Return total cost and average pass@1 per model/config."""
    if not runs:
        raise ValueError("No completed runs selected")
    if len({(run.dataset, run.repetitions) for run in runs.values()}) != 1:
        raise ValueError("Compared runs must use the same dataset and repetitions")
    task_categories: dict[str, str] | None = None
    records: list[dict[str, str | float]] = []
    for root, run in runs.items():
        scores = load_task_scores(run.path)
        current_tasks = {score.case_id: score.category for score in scores}
        if task_categories is None:
            task_categories = current_tasks
        elif current_tasks != task_categories:
            raise ValueError("Compared runs must contain the same tasks and categories")
        result = bootstrap_pass_rate(
            scores,
            tasks_per_category=tasks_per_category,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        )
        records.append(
            {
                "root": root,
                "total_cost_usd": _total_cost_usd(run),
                "pass_rate": result.estimate,
                "lower": result.confidence_interval[0],
                "upper": result.confidence_interval[1],
            }
        )
    data = pd.DataFrame.from_records(records)
    data["root"] = pd.Categorical(data["root"], categories=list(runs), ordered=True)
    return data


def make_pass_rate_cost_figure(
    data: pd.DataFrame, *, typography: ReportTypography = DEFAULT_TYPOGRAPHY
) -> ggplot:
    """Build a scatter plot with vertical pass@1 confidence intervals."""
    roots = list(data["root"].cat.categories)
    if len(roots) > len(COLORS):
        raise ValueError(f"Figure supports at most {len(COLORS)} model/config points")
    # Keep model colors and shapes consistent with the per-ablation figures.
    colors = dict(zip(sorted(roots), COLORS, strict=False))
    markers = dict(zip(sorted(roots), MARKERS, strict=False))
    minimum_cost = float(data["total_cost_usd"].min())
    maximum_cost = float(data["total_cost_usd"].max())
    cost_span = max(maximum_cost - minimum_cost, maximum_cost * 0.2, 1e-6)
    return (
        ggplot(
            data,
            aes(x="total_cost_usd", y="pass_rate", color="root", shape="root"),
        )
        + geom_errorbar(
            aes(ymin="lower", ymax="upper"), width=0.025 * cost_span, size=0.6
        )
        + geom_point(size=3.2)
        + geom_text(
            aes(label="root"),
            nudge_x=0.025 * cost_span,
            ha="left",
            va="center",
            size=typography.points,
            family=typography.font_family,
            color="black",
            fontweight="normal",
            show_legend=False,
        )
        + scale_color_manual(values=colors)
        + scale_shape_manual(values=markers)
        + scale_y_continuous(breaks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], expand=(0, 0))
        + coord_cartesian(
            xlim=(max(0, minimum_cost - 0.08 * cost_span), maximum_cost + 0.5 * cost_span),
            ylim=(0, 1),
        )
        + labs(
            x="Estimated total model cost (USD)",
            y="Average pass@1",
        )
        + theme_minimal(base_size=11)
        + theme(
            panel_grid_minor=element_blank(),
            panel_grid_major=element_line(color="#dddddd", size=0.35),
            panel_border=element_rect(color="#333333", fill=None, size=0.65),
            axis_text=element_text(color="#333333", size=12),
            axis_title=element_text(color="#222222", size=14),
            axis_ticks=element_line(color="#333333", size=0.65),
            axis_ticks_length=4,
            legend_position="none",
        )
        + typography_theme(typography)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--report-override", type=Path, action="append", default=[],
        help="Use this saved report version for its run; may be repeated",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ablation", help="Run-config suffix after '__'")
    selection.add_argument(
        "--best-ablation", action="store_true",
        help="Select each model's highest category-balanced overall pass@1 ablation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PDF path; defaults to output/pdf/pass_rate_vs_cost_<ablation|best>.pdf",
    )
    parser.add_argument("-m", "--tasks-per-category", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    add_figure_arguments(parser)
    args = parser.parse_args()
    typography = typography_from_args(args)

    completed = replace_run_reports(
        load_benchmark_runs(args.results_dir), args.report_override
    )
    if args.best_ablation:
        selected = select_best_ablation_runs(completed)
        model_rank = {name: index for index, name in enumerate(BEST_MODEL_ORDER)}
        selected.sort(key=lambda run: (
            model_rank.get(run.name.rpartition("__")[0], len(BEST_MODEL_ORDER)),
            run.name,
        ))
        runs = {run.name.rpartition("__")[0]: run for run in selected}
    else:
        runs = select_ablation_runs(completed, args.ablation)
    data = pass_rate_cost_data(
        runs,
        tasks_per_category=args.tasks_per_category,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )
    suffix = "best" if args.best_ablation else args.ablation
    output = args.output or DEFAULT_OUTPUT_DIR / f"pass_rate_vs_cost_{suffix}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = figure_dimensions(args, width_mm=120, height_mm=90)
    make_pass_rate_cost_figure(data, typography=typography).save(output, width=width, height=height, verbose=False)
    print(f"Wrote {output}")
    if args.copy_to_thesis:
        copy_figure_to_thesis(output, args.thesis_dir)


if __name__ == "__main__":
    main()
