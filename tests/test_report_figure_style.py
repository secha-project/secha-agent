from pathlib import Path

import pytest

from scripts.report_figure_style import (
    ReportTypography,
    copy_figure_to_thesis,
    natural_size_include,
)


def test_native_size_preserves_alt_text_and_unrelated_figures() -> None:
    source = (
        r"\includegraphics[alt={A plot, with intervals.},width=\linewidth]{images/results/plot.pdf}"
        "\n"
        r"\includegraphics[width=.6\linewidth]{images/results/other.pdf}"
    )
    updated = natural_size_include(source, "plot.pdf")
    assert updated == (
        r"\includegraphics[alt={A plot, with intervals.}]{images/results/plot.pdf}"
        "\n"
        r"\includegraphics[width=.6\linewidth]{images/results/other.pdf}"
    )
    assert natural_size_include(updated, "plot.pdf") == updated


def test_copy_keeps_pdf_bytes_and_updates_only_its_include(tmp_path: Path) -> None:
    thesis = tmp_path / "thesis"
    (thesis / "mainmatter").mkdir(parents=True)
    (thesis / "tauthesis.cls").write_text("test")
    chapter = thesis / "mainmatter/5-results.tex"
    chapter.write_text(r"\includegraphics[width=.8\linewidth]{images/results/plot.pdf}")
    source = tmp_path / "plot.pdf"
    source.write_bytes(b"test figure bytes")
    destination = copy_figure_to_thesis(source, thesis)
    assert destination.read_bytes() == source.read_bytes()
    assert chapter.read_text() == r"\includegraphics{images/results/plot.pdf}"


def test_font_size_matches_latex_points() -> None:
    assert ReportTypography(12).points == pytest.approx(12 * 72 / 72.27)
