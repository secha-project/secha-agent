import re
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import pandas as pd
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    EvaluatorOutput,
)

from ..eval_models import BenchmarkOutput, CaseMetadata, TableOutput
from ._numeric import numbers_match

_MISSING = object()
_ISO_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})?)?"
)


def _normalize_value(value):
    if pd.isna(value):
        return _MISSING
    if isinstance(value, str) and _ISO_TIMESTAMP.fullmatch(value):
        try:
            timestamp = pd.Timestamp(value)
            # Reference CSVs store timezone-free timestamps in UTC.
            if timestamp.tzinfo is None:
                return timestamp.tz_localize("UTC")
            return timestamp.tz_convert("UTC")
        except ValueError:
            pass
    return value


def _values_match(expected, actual, tolerance):
    if expected is _MISSING or actual is _MISSING:
        return expected is actual
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, Real) and isinstance(actual, Real):
        return numbers_match(expected, actual, tolerance)
    return expected == actual


def _column_positions(target_table, candidate_table):
    """Match names first, then pair remaining aliases in their column order."""
    if (
        len(candidate_table.columns) < len(target_table.columns)
        or not target_table.columns.is_unique
        or not candidate_table.columns.is_unique
    ):
        return None

    positions = {name: index for index, name in enumerate(candidate_table.columns)}
    matched = [positions.get(name) for name in target_table.columns]
    used = {index for index in matched if index is not None}
    remaining = iter(index for index in range(len(positions)) if index not in used)
    return [index if index is not None else next(remaining) for index in matched]


def _rows_match(expected, actual, tolerance):
    return all(
        _values_match(target, candidate, tolerance)
        for target, candidate in zip(expected, actual, strict=True)
    )


def _has_distinct_row_matches(matches, target_count):
    """Find one-to-one matches, reassigning earlier choices when tolerances overlap."""
    owners = [None] * target_count
    for start in range(len(matches)):
        parents = {start: None}
        pending = [start]
        end = None
        for candidate in pending:
            for target in matches[candidate]:
                owner = owners[target]
                if owner is None:
                    end = (candidate, target)
                    break
                if owner not in parents:
                    parents[owner] = (candidate, target)
                    pending.append(owner)
            if end is not None:
                break
        if end is None:
            return False

        # Flip the alternating path to free a target for this candidate row.
        while end is not None:
            candidate, target = end
            owners[target] = candidate
            end = parents[candidate]
    return True


def table_match_modes(
    target_table,
    candidate_table,
    partial=False,
    tolerance=0.005,
):
    """Return whether whole rows match in order and ignoring order, respectively."""
    positions = _column_positions(target_table, candidate_table)
    if positions is None or len(candidate_table) > len(target_table):
        return False, False
    if not partial and len(candidate_table) != len(target_table):
        return False, False

    target_rows = [
        tuple(_normalize_value(value) for value in row)
        for row in target_table.itertuples(index=False, name=None)
    ]
    candidate_rows = [
        tuple(_normalize_value(value) for value in row)
        for row in candidate_table.iloc[:, positions].itertuples(index=False, name=None)
    ]
    # An ordered partial match must be a subsequence of complete reference rows.
    remaining = iter(target_rows)
    if all(
        any(_rows_match(target, candidate, tolerance) for target in remaining)
        for candidate in candidate_rows
    ):
        return True, True

    matches = [
        [
            index
            for index, target in enumerate(target_rows)
            if _rows_match(target, candidate, tolerance)
        ]
        for candidate in candidate_rows
    ]
    return False, _has_distinct_row_matches(matches, len(target_rows))


def classify_table_match(
    target_table,
    candidate_table,
    min_coverage=0.5,
    tolerance=0.005,
):
    target_rows = len(target_table)
    candidate_rows = len(candidate_table)

    if target_rows == 0:
        if candidate_rows != 0:
            return "no_match", None

        ordered, unordered = table_match_modes(
            target_table,
            candidate_table,
            partial=False,
            tolerance=tolerance,
        )

        if ordered:
            return "exact_match", 1.0
        if unordered:
            return "match_up_to_ordering", 1.0
        return "no_match", None

    coverage = candidate_rows / target_rows

    if candidate_rows > target_rows:
        return "no_match", coverage

    partial = candidate_rows < target_rows

    if partial and coverage < min_coverage:
        return "no_match", coverage

    ordered, unordered = table_match_modes(
        target_table,
        candidate_table,
        partial=partial,
        tolerance=tolerance,
    )

    if ordered:
        match_type = "partial_exact_match" if partial else "exact_match"
        return match_type, coverage

    if unordered:
        match_type = (
            "partial_match_up_to_ordering" if partial else "match_up_to_ordering"
        )
        return match_type, coverage

    return "no_match", coverage


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
