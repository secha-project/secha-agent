"""Show category accuracies inside the original ablation bars.

The stacked layout divides each category's pass@1 by the number of categories.
These equal-weight contributions sum to the original stratified overall pass@1;
they are not the unscaled category accuracies. Shades identify categories in a
fixed bottom-to-top order, while hue preserves the original ablation colors.
Whiskers describe the overall estimate, not individual stack segments.

Use the faceted layout to compare actual category accuracies on a shared scale;
its whiskers describe category estimates. Both layouts use percentile bootstrap
intervals, with the same options as generate_ablation_figure.py.

Examples:
    python -m scripts.generate_ablation_category_figure -m 20
    python -m scripts.generate_ablation_category_figure -m 20 --layout faceted

A CSV beside the figure records raw category rates, weights, contributions,
and both category and overall confidence intervals. No reports are modified.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_cartesian,
    element_blank,
    element_line,
    element_rect,
    element_text,
    facet_wrap,
    geom_col,
    geom_errorbar,
    geom_rect,
    ggplot,
    guide_legend,
    guides,
    labs,
    position_dodge,
    scale_alpha_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_x_discrete,
    scale_y_continuous,
    theme,
    theme_minimal,
)

from scripts.bootstrap_pass_rate import TaskScore, bootstrap_pass_rate, load_task_scores
from scripts.generate_ablation_figure import COLORS, PROJECT_ROOT, ablation_data
from scripts.generate_pass_rate_table import CATEGORY_ORDER
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

OUTPUT_DIR = PROJECT_ROOT / "output/pdf"


def ablation_category_data(
    runs: Sequence[BenchmarkRun],
    *,
    tasks_per_category: int,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    roots: Sequence[str] = (),
) -> pd.DataFrame:
    """Return tidy category estimates after the original compatibility checks."""
    options = {
        "tasks_per_category": tasks_per_category,
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
        "seed": seed,
    }
    overall = ablation_data(runs, roots=roots, **options)
    by_name = {run.name: run for run in runs}
    records = []
    for row in overall.itertuples(index=False):
        scores = load_task_scores(by_name[f"{row.root}__{row.ablation}"].path)
        grouped: dict[str, list[TaskScore]] = defaultdict(list)
        for score in scores:
            grouped[score.category].append(score)
        weight = 1 / len(grouped)
        for category, category_scores in grouped.items():
            estimate = bootstrap_pass_rate(category_scores, **options)
            records.append(
                {
                    "root": row.root,
                    "ablation": row.ablation,
                    "category": category,
                    "tasks": len(category_scores),
                    "pass_rate": estimate.estimate,
                    "lower": estimate.confidence_interval[0],
                    "upper": estimate.confidence_interval[1],
                    "weight": weight,
                    "contribution": weight * estimate.estimate,
                    "overall_pass_rate": row.pass_rate,
                    "overall_lower": row.lower,
                    "overall_upper": row.upper,
                }
            )

    data = pd.DataFrame.from_records(records)
    for column in ("root", "ablation"):
        data[column] = pd.Categorical(
            data[column], categories=overall[column].cat.categories, ordered=True
        )
    names = set(data["category"])
    categories = [name for name in CATEGORY_ORDER if name in names]
    categories.extend(sorted(names - set(categories)))
    data["category"] = pd.Categorical(
        data["category"], categories=categories, ordered=True
    )
    return data.sort_values(["root", "ablation", "category"]).reset_index(drop=True)


def _fill_scale(data: pd.DataFrame):
    ablations = list(data["ablation"].cat.categories)
    # Match the source figure's color assignment, including its legacy order.
    color_order = sorted(ablations, key=lambda value: (value != "baseline", value))
    colors = {name: COLORS[i % len(COLORS)] for i, name in enumerate(color_order)}
    return scale_fill_manual(
        values=colors,
        breaks=ablations,
        labels=[name.replace("_", " ").replace("-", " ").title().replace("Few Shot Semantic Layer", "Few Shot\nSemantic Layer") for name in ablations],
    )


def _figure_theme(typography: ReportTypography = DEFAULT_TYPOGRAPHY):
    return theme_minimal(base_size=11) + theme(
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
        strip_text=element_text(color="#222222", size=12),
        panel_spacing=0.045,
    ) + typography_theme(typography)


def make_stacked_ablation_figure(
    data: pd.DataFrame, *, typography: ReportTypography = DEFAULT_TYPOGRAPHY
) -> ggplot:
    """Build weighted category stacks; their heights equal overall pass@1."""
    segments = data.copy()
    roots = list(data["root"].cat.categories)
    ablations = list(data["ablation"].cat.categories)
    categories = list(data["category"].cat.categories)
    slot_width = 0.8 / len(ablations)
    segments["x"] = (
        segments["root"].cat.codes
        + 1
        + (segments["ablation"].cat.codes - (len(ablations) - 1) / 2) * slot_width
    )
    segments["xmin"] = segments["x"] - 0.45 * slot_width
    segments["xmax"] = segments["x"] + 0.45 * slot_width
    segments = segments.sort_values(["root", "ablation", "category"])
    segments["ymax"] = segments.groupby(["root", "ablation"], observed=True)[
        "contribution"
    ].cumsum()
    segments["ymin"] = segments["ymax"] - segments["contribution"]
    bars = segments.drop_duplicates(["root", "ablation"]).copy()
    bars["ymin"] = 0.0
    bars["ymax"] = bars["overall_pass_rate"]
    intensities = dict(zip(categories, np.linspace(0.35, 1.0, len(categories))))
    return (
        ggplot()
        # White backing makes alpha a white tint, independent of gridlines.
        + geom_rect(
            bars,
            aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
            fill="white",
            color=None,
            show_legend=False,
        )
        + geom_rect(
            segments,
            aes(
                xmin="xmin",
                xmax="xmax",
                ymin="ymin",
                ymax="ymax",
                fill="ablation",
                alpha="category",
            ),
            color=None,
        )
        + geom_rect(
            segments,
            aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
            fill=None,
            color="white",
            size=0.3,
            show_legend=False,
        )
        + geom_rect(
            bars,
            aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
            fill=None,
            color="#333333",
            size=0.35,
            show_legend=False,
        )
        + geom_errorbar(
            bars,
            aes(x="x", ymin="overall_lower", ymax="overall_upper"),
            width=0.65 * slot_width,
            color="#252525",
            size=0.55,
            show_legend=False,
        )
        + _fill_scale(data)
        + scale_alpha_manual(
            values=intensities,
            breaks=categories,
            labels=[name.capitalize() for name in categories],
        )
        + guides(
            fill=guide_legend(order=1, override_aes={"alpha": 1}),
            alpha=guide_legend(order=2, override_aes={"fill": "#555555"}),
        )
        + scale_x_continuous(
            breaks=list(range(1, len(roots) + 1)),
            labels=roots,
            limits=(0.5, len(roots) + 0.5),
            expand=(0, 0),
        )
        + scale_y_continuous(breaks=np.linspace(0, 1, 6), expand=(0, 0))
        + coord_cartesian(ylim=(0, 1))
        + labs(
            x="",
            y="Pass@1 (equal category weights)",
            fill="Ablation",
            alpha="Category (bottom to top)",
        )
        + _figure_theme(typography)
    )


def make_faceted_ablation_figure(
    data: pd.DataFrame, *, legend_position: str = "top",
    typography: ReportTypography = DEFAULT_TYPOGRAPHY,
) -> ggplot:
    """Compare raw category pass@1 with category-specific bootstrap intervals."""
    panels = data.copy()
    panels["category"] = panels["category"].cat.rename_categories(str.capitalize)
    dodge = position_dodge(width=0.8)
    return (
        ggplot(panels, aes(x="root", y="pass_rate", fill="ablation"))
        + geom_col(position=dodge, width=0.72, color="#333333", size=0.25)
        + geom_errorbar(
            aes(ymin="lower", ymax="upper"),
            position=dodge,
            width=0.13,
            color="#252525",
            size=0.40,
        )
        # Repeat tick labels in every panel. Explicit limits keep the scales
        # identical even though facet_wrap needs "free" to draw all axes.
        + facet_wrap("category", ncol=2, scales="free")
        + _fill_scale(data)
        + scale_x_discrete(
            limits=list(data["root"].cat.categories),
            labels=lambda values: [name.replace("gpt-6-", "gpt-6\n").replace("qwen3.8-", "qwen3.8\n") for name in values],
        )
        + scale_y_continuous(limits=(0, 1), breaks=np.linspace(0, 1, 6), expand=(0, 0))
        + coord_cartesian(ylim=(0, 1))
        + labs(x="", y="Category pass@1", fill="Ablation")
        + _figure_theme(typography)
        + guides(fill=guide_legend(nrow=1, direction="horizontal"))
        + theme(
            legend_position=legend_position,
            legend_direction="horizontal",
            legend_justification="center",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layout", choices=("stacked", "faceted"), default="stacked")
    parser.add_argument(
        "--legend-position",
        choices=("top", "bottom"),
        default="top",
        help="Position of the horizontal legend in the faceted layout.",
    )
    parser.add_argument(
        "--root", action="append", default=[], help="Run root; repeatable."
    )
    parser.add_argument("-m", "--tasks-per-category", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    add_figure_arguments(parser)
    args = parser.parse_args()
    typography = typography_from_args(args)
    data = ablation_category_data(
        load_benchmark_runs(args.results_dir),
        tasks_per_category=args.tasks_per_category,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
        roots=args.root,
    )
    output = args.output or OUTPUT_DIR / f"ablation_category_{args.layout}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.layout == "stacked":
        figure = make_stacked_ablation_figure(data, typography=typography)
        width, height = figure_dimensions(args, width_mm=150, height_mm=110)
    else:
        figure = make_faceted_ablation_figure(data, legend_position=args.legend_position, typography=typography)
        width, height = figure_dimensions(args, width_mm=150, height_mm=145)
    figure.save(output, width=width, height=height, verbose=False)
    csv_output = output.with_suffix(".csv")
    data.to_csv(csv_output, index=False)
    print(f"Wrote {output}")
    print(f"Wrote {csv_output}")
    if args.copy_to_thesis:
        copy_figure_to_thesis(output, args.thesis_dir)


if __name__ == "__main__":
    main()
