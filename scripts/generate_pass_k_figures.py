"""Plot pass@k and pass^k for all completed runs of one ablation.

Run configs must be named ROOT__ABLATION, for example model-a__baseline.
Both metrics are shown as stacked plots, each with its own y-axis label,
on matching 0-1 scales.

Usage: python -m scripts.generate_pass_k_figures --ablation baseline
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
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    scale_shape_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
)
from plotnine.composition import Compose

from scripts.generate_ablation_figure import split_ablation_name
from scripts.load_benchmark_runs import (
    DEFAULT_RESULTS_DIR,
    BenchmarkRun,
    load_benchmark_runs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/pdf"
METRICS = (("pass_at_k", "pass@k"), ("pass_power_k", "pass^k"))
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
    """Select one completed run per root for the requested suffix."""
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
    if next(iter(selected.values())).repetitions < 2:
        raise ValueError("At least two repetitions are needed to draw pass-k curves")
    return dict(sorted(selected.items()))


def _task_categories(run: BenchmarkRun) -> dict[str, str]:
    return {
        case_id: category
        for category, result in run.results["categories"].items()
        for case_id in result["tasks"]
    }


def _metric_values(run: BenchmarkRun, key: str) -> dict[int, float]:
    values = run.results["overall"].get(key)
    if values is None:
        raise ValueError(f"Run {run.name!r} has no overall {key} metrics")
    if not isinstance(values, dict):
        raise TypeError(f"Run {run.name!r} has invalid overall {key} metrics")
    parsed: dict[int, float] = {}
    for raw_k, value in values.items():
        if not isinstance(raw_k, str) or not raw_k.isdecimal():
            raise ValueError(f"Invalid k in {run.name!r} overall {key}: {raw_k!r}")
        k = int(raw_k)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Invalid {run.name!r} overall {key} value for k={k}")
        parsed[k] = float(value)
    expected = set(range(1, run.repetitions + 1))
    if parsed.keys() != expected:
        raise ValueError(
            f"Run {run.name!r} needs overall {key} for every k from 1 to {run.repetitions}"
        )
    return parsed


def pass_k_data(runs: dict[str, BenchmarkRun]) -> pd.DataFrame:
    """Return tidy saved metrics without converting their 0-1 scale."""
    task_categories: dict[str, str] | None = None
    records: list[dict[str, str | int | float]] = []
    for root, run in runs.items():
        current_tasks = _task_categories(run)
        if task_categories is None:
            task_categories = current_tasks
        elif current_tasks != task_categories:
            raise ValueError("Compared runs must contain the same tasks and categories")
        for key, label in METRICS:
            records.extend(
                {"root": root, "metric": label, "k": k, "score": score}
                for k, score in sorted(_metric_values(run, key).items())
            )

    data = pd.DataFrame.from_records(records)
    data["root"] = pd.Categorical(data["root"], categories=list(runs), ordered=True)
    data["metric"] = pd.Categorical(
        data["metric"], categories=[label for _, label in METRICS], ordered=True
    )
    return data


def make_pass_k_figure(data: pd.DataFrame) -> Compose:
    """Stack two separately labeled plots with matching k and score scales."""
    roots = list(data["root"].cat.categories)
    if len(roots) > len(COLORS):
        raise ValueError(f"Figure supports at most {len(COLORS)} model/config lines")
    colors = dict(zip(roots, COLORS, strict=False))
    markers = dict(zip(roots, MARKERS, strict=False))
    maximum_k = int(data["k"].max())
    breaks = (
        list(range(1, maximum_k + 1))
        if maximum_k <= 10
        else sorted({1, maximum_k, *range(5, maximum_k + 1, 5)})
    )

    def plot_metric(label: str) -> ggplot:
        metric_data = data[data["metric"] == label]
        return (
            ggplot(
                metric_data,
                aes(x="k", y="score", color="root", shape="root", group="root"),
            )
            + geom_line(size=0.85)
            + geom_point(data=metric_data[metric_data["k"].isin(breaks)], size=2.5)
            + scale_color_manual(values=colors)
            + scale_shape_manual(values=markers)
            + scale_x_continuous(breaks=breaks, minor_breaks=[])
            + scale_y_continuous(breaks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], expand=(0, 0))
            + coord_cartesian(xlim=(0.8, maximum_k + 0.2), ylim=(0, 1))
            + labs(
                x="k",
                y="Pass@k" if label == "pass@k" else "Pass^k",
                color="Model/config",
                shape="Model/config",
            )
            + theme_bw(base_size=11)
            + theme(
                panel_grid_major=element_line(color="#dddddd", size=0.35),
                panel_grid_minor=element_blank(),
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
        )

    top = plot_metric("pass@k") + theme(
        axis_title_x=element_blank(),
        axis_text_x=element_blank(),
        axis_ticks_x=element_blank(),
    )
    bottom = plot_metric("pass^k") + theme(legend_position="none")
    return top / bottom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--ablation", required=True, help="Run-config suffix after '__'"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PDF path; defaults to output/pdf/pass_k_<ablation>.pdf",
    )
    args = parser.parse_args()

    runs = select_ablation_runs(load_benchmark_runs(args.results_dir), args.ablation)
    data = pass_k_data(runs)
    output = args.output or DEFAULT_OUTPUT_DIR / f"pass_k_{args.ablation}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    width = max(7.5, 1.15 * len(runs) + 3.0)
    make_pass_k_figure(data).save(output, width=width, height=7.4, verbose=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
