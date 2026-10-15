"""CSV evaluator entry points and table match classification.

Matching algorithms live in _table_matching; cell normalization and rounding
rules live in _table_values. Column names are arbitrary SQL aliases.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput

from ..eval_models import BenchmarkOutput, CaseMetadata, TableOutput

# Existing benchmark audit scripts import these helpers from this module.
from ._table_matching import _column_positions, table_match_modes  # noqa: F401
from ._table_values import (  # noqa: F401
    _normalize_value,
    _rounded_column,
    _values_match,
)

_EXPLICIT_PRECISION = re.compile(
    r"\bunrounded\b|\b(?:full|original)\s+precision\b"
    r"|\b(?:without|no)\s+rounding\b|\b(?:do not|don't)\s+round\b"
    r"|\b(?:decimal|significant)\s+(?:places?|figures?|digits?)\b"
    r"|\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)[ -]decimals?\b",
    re.IGNORECASE,
)


def classify_table_match(
    target_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    min_coverage: float = 0.5,
    tolerance: float = 0.005,
    minimum_significant_digits: int | None = None,
) -> tuple[str, float | None]:
    """Classify full or partial matches; partial matches remain diagnostic only."""
    target_rows = len(target_table)
    candidate_rows = len(candidate_table)
    coverage = candidate_rows / target_rows if target_rows else None
    partial = candidate_rows < target_rows
    if candidate_rows > target_rows or (
        partial and coverage is not None and coverage < min_coverage
    ):
        return "no_match", coverage

    modes = table_match_modes(
        target_table,
        candidate_table,
        partial=partial,
        tolerance=tolerance,
        minimum_significant_digits=minimum_significant_digits,
    )

    if not modes.unordered:
        return "no_match", coverage
    match_type = "exact_match" if modes.ordered else "match_up_to_ordering"
    if partial:
        match_type = f"partial_{match_type}"
    return match_type, coverage if coverage is not None else 1.0


def _table_outputs(expected_output):
    if not isinstance(expected_output, list):
        return []

    return [item for item in expected_output if isinstance(item, TableOutput)]


@dataclass
class ValidateTableIncluded(Evaluator[str, BenchmarkOutput, CaseMetadata]):
    """Check that the agent selected a table to return to the user."""

    def evaluate(
        self, ctx: EvaluatorContext[str, BenchmarkOutput, CaseMetadata]
    ) -> EvaluatorOutput:
        if not _table_outputs(ctx.expected_output):
            return {}

        return "csv_path" in ctx.attributes


@dataclass
class ValidateTableValues(Evaluator[str, BenchmarkOutput, CaseMetadata]):
    """Compare the selected table with every expected table."""

    artifact_base_path: Path = Path()
    min_coverage: float = 0.5
    tolerance: float = 0.005
    minimum_significant_digits: int = 2

    def __post_init__(self):
        if not 2 <= self.minimum_significant_digits <= 15:
            raise ValueError("Table rounding requires 2 to 15 significant digits")

    def evaluate(
        self, ctx: EvaluatorContext[str, BenchmarkOutput, CaseMetadata]
    ) -> EvaluatorOutput:
        table_outputs = _table_outputs(ctx.expected_output)
        if not table_outputs:
            return {}

        csv_path = ctx.attributes.get("csv_path")
        if not isinstance(csv_path, (str, Path)):
            return {
                "ValidateTableValues": False,
                "TableMatchType": "no_match",
            }

        candidate_table = pd.read_csv(csv_path)
        overall_match_type = "exact_match"
        inputs = getattr(ctx, "inputs", "")
        rounding = (
            None
            if isinstance(inputs, str) and _EXPLICIT_PRECISION.search(inputs)
            else self.minimum_significant_digits
        )

        for table_output in table_outputs:
            target_path = Path(table_output.table_path)
            if not target_path.is_absolute():
                target_path = self.artifact_base_path / target_path

            target_table = pd.read_csv(target_path)
            match_type, _ = classify_table_match(
                target_table,
                candidate_table,
                min_coverage=self.min_coverage,
                tolerance=self.tolerance,
                minimum_significant_digits=rounding,
            )
            if match_type != "exact_match" and not (
                match_type == "match_up_to_ordering" and not table_output.ordered
            ):
                return {
                    "ValidateTableValues": False,
                    "TableMatchType": match_type,
                }

            if match_type == "match_up_to_ordering":
                overall_match_type = match_type

        return {
            "ValidateTableValues": True,
            "TableMatchType": overall_match_type,
        }
