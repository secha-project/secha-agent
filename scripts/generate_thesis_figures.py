"""Generate and copy the benchmark figures used in the thesis results chapter.

Usage: python -m scripts.generate_thesis_figures
       python -m scripts.generate_thesis_figures --figure cost --width-mm 130 --height-mm 85

Figure text defaults to 10 LaTeX pt regardless of the PDF dimensions. Figures
are included at their natural sizes; the copy step removes their LaTeX resizing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.generate_ablation_category_figure import (
    ablation_category_data,
    make_faceted_ablation_figure,
)
from scripts.generate_ablation_figure import ablation_data, make_ablation_figure
from scripts.generate_pass_k_figures import make_pass_k_figure, pass_k_data
from scripts.generate_pass_rate_cost_figure import (
    make_pass_rate_cost_figure,
    pass_rate_cost_data,
    replace_run_reports,
)
from scripts.generate_pass_rate_table import BEST_MODEL_ORDER, select_best_ablation_runs
from scripts.load_benchmark_runs import DEFAULT_RESULTS_DIR, load_benchmark_runs
from scripts.report_figure_style import (
    PROJECT_ROOT,
    add_figure_arguments,
    copy_figure_to_thesis,
    figure_dimensions,
    save_report_figure,
    typography_from_args,
)

FIGURES = {
    "ablation": ("ablation_pass_rates.pdf", 150, 95),
    "categories": ("ablation_category_faceted_bottom.pdf", 150, 145),
    "pass-k": ("pass_k_best.pdf", 150, 80),
    "cost": ("pass_rate_vs_cost_best.pdf", 120, 90),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=FIGURES,
        action="append",
        help="Regenerate only this figure; repeatable (default: all four)",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--dataset", default="power-quality-monitoring")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output/pdf")
    parser.add_argument("-m", "--tasks-per-category", type=int, default=15)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-override", type=Path, action="append", default=[])
    add_figure_arguments(parser)
    parser.add_argument(
        "--no-copy",
        action="store_false",
        dest="copy_to_thesis",
        help="Generate locally without changing the thesis",
    )
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Copy already-generated PDFs without regenerating them",
    )
    parser.set_defaults(copy_to_thesis=True)
    args = parser.parse_args()
    figures = list(dict.fromkeys(args.figure or FIGURES))
    if (args.width_mm or args.height_mm) and len(figures) != 1:
        parser.error("Select a single --figure when overriding width or height")
    if args.copy_only:
        if not args.copy_to_thesis:
            parser.error("--copy-only cannot be combined with --no-copy")
        for key in figures:
            copy_figure_to_thesis(args.output_dir / FIGURES[key][0], args.thesis_dir)
        return

    typography = typography_from_args(args)
    runs = [
        run
        for run in load_benchmark_runs(args.results_dir)
        if run.dataset == args.dataset
        and run.name.rpartition("__")[0] in BEST_MODEL_ORDER
    ]
    # Prefer completed aggregations saved beside the original reports.
    completed_paths = [
        run.path.with_name("report-complete.json")
        for run in runs
        if run.path.with_name("report-complete.json").is_file()
    ]
    runs = replace_run_reports(runs, completed_paths)
    runs = replace_run_reports(runs, args.report_override)
    selected = {
        run.name.rpartition("__")[0]: run for run in select_best_ablation_runs(runs)
    }
    best = {root: selected[root] for root in BEST_MODEL_ORDER}
    options = {
        "tasks_per_category": args.tasks_per_category,
        "bootstrap_samples": args.bootstrap_samples,
        "confidence_level": args.confidence_level,
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in figures:
        filename, default_width, default_height = FIGURES[key]
        if key == "ablation":
            data = ablation_data(runs, **options)
            figure = make_ablation_figure(data, typography=typography)
        elif key == "categories":
            data = ablation_category_data(runs, **options)
            figure = make_faceted_ablation_figure(
                data, legend_position="bottom", typography=typography
            )
            data.to_csv((args.output_dir / filename).with_suffix(".csv"), index=False)
        elif key == "pass-k":
            figure = make_pass_k_figure(pass_k_data(best), typography=typography)
        else:
            figure = make_pass_rate_cost_figure(
                pass_rate_cost_data(best, **options), typography=typography
            )
        width, height = figure_dimensions(
            args, width_mm=default_width, height_mm=default_height
        )
        output = args.output_dir / filename
        save_report_figure(figure, output, width=width, height=height)
        print(f"Wrote {output} ({width * 25.4:g} x {height * 25.4:g} mm)")
        if args.copy_to_thesis:
            copy_figure_to_thesis(output, args.thesis_dir)


if __name__ == "__main__":
    main()
