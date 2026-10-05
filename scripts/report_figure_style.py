"""Shared thesis typography, physical figure dimensions, and artifact copying."""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from matplotlib import font_manager
from plotnine import element_text, ggplot, theme
from plotnine.composition import Compose

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THESIS_DIR = PROJECT_ROOT.parent.parent / "tuni-masters-thesis"
MM_PER_INCH = 25.4
LATEX_PT_TO_PDF_PT = 72 / 72.27


@dataclass(frozen=True)
class ReportTypography:
    font_size: float = 10
    font_family: str = "Roboto"

    @property
    def points(self) -> float:
        """Matplotlib uses PDF points; the thesis specifies LaTeX points."""
        return self.font_size * LATEX_PT_TO_PDF_PT


DEFAULT_TYPOGRAPHY = ReportTypography()


@lru_cache
def register_thesis_fonts(thesis_dir: Path) -> None:
    directory = thesis_dir / "fonts/Roboto"
    for variant in ("Regular", "Bold", "Italic", "BoldItalic"):
        path = directory / f"Roboto-{variant}.ttf"
        if not path.is_file():
            raise FileNotFoundError(f"Thesis font not found: {path}")
        font_manager.fontManager.addfont(path)


def typography_theme(typography: ReportTypography = DEFAULT_TYPOGRAPHY) -> theme:
    """Use the same physical font size for every kind of plot text."""
    text = element_text(family=typography.font_family, size=typography.points)
    return theme(
        text=text,
        axis_text=text,
        axis_title=text,
        legend_text=element_text(
            family=typography.font_family, size=typography.points, ma="left"
        ),
        legend_title=text,
        strip_text=text,
        plot_title=text,
        plot_subtitle=text,
        plot_caption=text,
    )


def positive_number(value: str) -> float:
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def add_figure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--width-mm", type=positive_number, help="PDF width in millimetres"
    )
    parser.add_argument(
        "--height-mm", type=positive_number, help="PDF height in millimetres"
    )
    parser.add_argument(
        "--font-size",
        type=positive_number,
        default=DEFAULT_TYPOGRAPHY.font_size,
        help="Font size in LaTeX points (default: 10)",
    )
    parser.add_argument("--font-family", default="Roboto")
    parser.add_argument("--thesis-dir", type=Path, default=DEFAULT_THESIS_DIR)
    parser.add_argument(
        "--copy-to-thesis",
        action="store_true",
        help="Copy the generated PDF into the thesis images/results directory",
    )


def typography_from_args(args: argparse.Namespace) -> ReportTypography:
    if args.font_family == "Roboto":
        register_thesis_fonts(args.thesis_dir.resolve())
    return ReportTypography(args.font_size, args.font_family)


def figure_dimensions(
    args: argparse.Namespace, *, width_mm: float, height_mm: float
) -> tuple[float, float]:
    return (
        (args.width_mm or width_mm) / MM_PER_INCH,
        (args.height_mm or height_mm) / MM_PER_INCH,
    )


def save_report_figure(
    figure: ggplot | Compose, output: Path, *, width: float, height: float
) -> None:
    # Compose.save ignores width/height keyword arguments. Its canvas dimensions
    # come from the last plot's figure_size theme instead.
    size_theme = theme(figure_size=(width, height))
    sized = figure & size_theme if isinstance(figure, Compose) else figure + size_theme
    sized.save(output, width=width, height=height, verbose=False)


def copy_figure_to_thesis(output: Path, thesis_dir: Path) -> Path:
    if not (thesis_dir / "tauthesis.cls").is_file():
        raise ValueError(f"Not a thesis project: {thesis_dir}")
    destination = thesis_dir / "images/results" / output.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() != destination.resolve():
        shutil.copy2(output, destination)
    chapter = thesis_dir / "mainmatter/5-results.tex"
    if chapter.is_file():
        original = chapter.read_text(encoding="utf-8")
        updated = natural_size_include(original, output.name)
        if updated != original:
            chapter.write_text(updated, encoding="utf-8")
            print(f"Updated {chapter}: {output.name} uses its generated size")
    print(f"Copied {destination}")
    return destination


def natural_size_include(source: str, filename: str) -> str:
    """Remove LaTeX resizing only for the generated figure being copied."""
    pattern = re.compile(
        r"(\\includegraphics)(?:\[([^\]]*)\])?"
        r"(\{images/results/" + re.escape(filename) + r"\})"
    )

    def replace(match: re.Match[str]) -> str:
        options = re.sub(
            r"(?:^|,)\s*(?:width|height|scale)\s*=[^,]*",
            "",
            match[2] or "",
        ).strip(" ,")
        return match[1] + (f"[{options}]" if options else "") + match[3]

    return pattern.sub(replace, source)
