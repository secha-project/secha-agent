"""Plot Figure 1 from completed runs named ROOT__ABLATION.

For example, openai-mini__baseline and openai-mini__semantic_layer form
one group. Ablation suffixes are discovered from the reports, not hard-coded.

Usage: python -m scripts.generate_ablation_figure -m 20
"""

from __future__ import annotations

import argparse
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
    geom_col,
    geom_errorbar,
    ggplot,
    labs,
    position_dodge,
    scale_fill_manual,
    scale_x_discrete,
    scale_y_continuous,
    theme,
    theme_minimal,
)

from scripts.bootstrap_pass_rate import bootstrap_pass_rate, load_task_scores
from scripts.load_benchmark_runs import (
    DEFAULT_RESULTS_DIR,
    BenchmarkRun,
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
DEFAULT_OUTPUT = PROJECT_ROOT / "output/pdf/ablation_pass_rates.pdf"
COLORS = ("#72b2a2", "#e99778", "#94a3c3", "#d4b46d", "#aa91b5")
ABLATION_ORDER = (
    "baseline",
    "few_shot",
    "semantic_layer",
    "few_shot_semantic_layer",
)
MODEL_ORDER = ("qwen3.8-27b", "gpt-6-luna", "glm-5.3", "gpt-6-sol")


def split_ablation_name(name: str) -> tuple[str, str] | None:
    """Return the run root and ablation suffix, if the naming rule applies."""
    root, separator, ablation = name.rpartition("__")
    if not separator or not root or not ablation:
        return None
    return root, ablation


def ablation_data(
    runs: Sequence[BenchmarkRun],
    *,
    tasks_per_category: int,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    roots: Sequence[str] = (),
) -> pd.DataFrame:
    """Return one pass@1 estimate and stratified CI per root/ablation."""
    selected_roots = set(roots)
    grouped: dict[str, dict[str, BenchmarkRun]] = {}
    for run in runs:
        parsed = split_ablation_name(run.name)
        if parsed is None:
            continue
        root, ablation = parsed
        if selected_roots and root not in selected_roots:
            continue
        variants = grouped.setdefault(root, {})
        if ablation in variants:
            raise ValueError(f"Duplicate run for {root}__{ablation}")
        variants[ablation] = run

    if not grouped:
        raise ValueError("No completed ROOT__ABLATION runs were found")
    missing_roots = selected_roots - grouped.keys()
    if missing_roots:
        raise ValueError(f"No completed runs for roots: {sorted(missing_roots)}")

    ablations = {ablation for variants in grouped.values() for ablation in variants}
    if len(ablations) < 2:
        raise ValueError("Figure 1 needs at least two ablation suffixes")
    for root, variants in grouped.items():
        missing = ablations - variants.keys()
        if missing:
            raise ValueError(f"{root} is missing ablations: {sorted(missing)}")
        identities = {
            (run.model_provider, run.model_name, run.dataset, run.repetitions)
            for run in variants.values()
        }
        if len(identities) != 1:
            raise ValueError(
                f"Runs under {root} must use the same model, dataset and repetitions"
            )

    all_runs = [run for variants in grouped.values() for run in variants.values()]
    if len({run.dataset for run in all_runs}) != 1:
        raise ValueError("Figure 1 runs must use the same dataset")

    ablation_rank = {name: index for index, name in enumerate(ABLATION_ORDER)}
    ordered_ablations = sorted(
        ablations, key=lambda value: (ablation_rank.get(value, len(ABLATION_ORDER)), value)
    )
    model_rank = {name: index for index, name in enumerate(MODEL_ORDER)}
    ordered_roots = sorted(
        grouped, key=lambda value: (model_rank.get(value, len(MODEL_ORDER)), value)
    )
    task_categories: dict[str, str] | None = None
    records: list[dict[str, str | float]] = []
    for root in ordered_roots:
        for ablation in ordered_ablations:
            run = grouped[root][ablation]
            scores = load_task_scores(run.path)
            current_tasks = {score.case_id: score.category for score in scores}
            if task_categories is None:
                task_categories = current_tasks
            elif current_tasks != task_categories:
                raise ValueError(
                    "Figure 1 runs must contain the same tasks and categories"
                )
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
                    "ablation": ablation,
                    "pass_rate": result.estimate,
                    "lower": result.confidence_interval[0],
                    "upper": result.confidence_interval[1],
                }
            )

    data = pd.DataFrame.from_records(records)
    data["root"] = pd.Categorical(
        data["root"], categories=ordered_roots, ordered=True
    )
    data["ablation"] = pd.Categorical(
        data["ablation"], categories=ordered_ablations, ordered=True
    )
    return data


def make_ablation_figure(
    data: pd.DataFrame, *, typography: ReportTypography = DEFAULT_TYPOGRAPHY
) -> ggplot:
    """Build a grouped bar plot with aligned percentile-bootstrap intervals."""
    ablations = list(data["ablation"].cat.categories)
    # Preserve existing colors while changing the bars' and legend's order.
    color_order = sorted(ablations, key=lambda value: (value != "baseline", value))
    colors = {name: COLORS[index % len(COLORS)] for index, name in enumerate(color_order)}
    labels = [name.replace("_", " ").replace("-", " ").title().replace("Few Shot Semantic Layer", "Few Shot\nSemantic Layer") for name in ablations]
    dodge = position_dodge(width=0.8)
    return (
        ggplot(data, aes(x="root", y="pass_rate", fill="ablation"))
        + geom_col(position=dodge, width=0.72, color="#333333", size=0.35)
        + geom_errorbar(
            aes(ymin="lower", ymax="upper"),
            position=dodge,
            width=0.13,
            color="#252525",
            size=0.55,
        )
        + scale_fill_manual(values=colors, breaks=ablations, labels=labels)
        + scale_x_discrete(labels=lambda values: [name.replace("gpt-6-", "gpt-6\n").replace("qwen3.8-", "qwen3.8\n") for name in values])
        + scale_y_continuous(breaks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], expand=(0, 0))
        + coord_cartesian(ylim=(0, 1))
        + labs(x="", y="Pass@1", fill="Ablation")
        + theme_minimal(base_size=11)
        + theme(
            panel_grid_major_x=element_blank(),
            panel_grid_minor=element_blank(),
            panel_grid_major_y=element_line(color="#dddddd", size=0.35),
            panel_border=element_rect(color="#333333", fill=None, size=0.65),
            axis_text=element_text(color="#333333", size=12),
            axis_title=element_text(color="#222222", size=14),
            axis_ticks=element_line(color="#333333", size=0.65),
            axis_ticks_length=4,
            legend_title=element_text(color="#222222", size=12),
            legend_text=element_text(color="#222222", size=11),
            legend_position="right",
            legend_justification_right="top",
        )
        + typography_theme(typography)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Include only this run root; repeatable.",
    )
    parser.add_argument("-m", "--tasks-per-category", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    add_figure_arguments(parser)
    args = parser.parse_args()
    typography = typography_from_args(args)

    data = ablation_data(
        load_benchmark_runs(args.results_dir),
        tasks_per_category=args.tasks_per_category,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
        roots=args.root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    width, height = figure_dimensions(args, width_mm=150, height_mm=95)
    make_ablation_figure(data, typography=typography).save(args.output, width=width, height=height, verbose=False)
    print(f"Wrote {args.output}")
    if args.copy_to_thesis:
        copy_figure_to_thesis(args.output, args.thesis_dir)


if __name__ == "__main__":
    main()
