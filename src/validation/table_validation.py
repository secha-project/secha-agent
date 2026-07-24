from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    EvaluatorOutput,
)

from .eval_models import BenchmarkOutput, CaseMetadata, TableOutput

_MISSING = object()


def _value_key(value):
    return _MISSING if pd.isna(value) else value


def _column_counts(column):
    return Counter(_value_key(value) for value in column)


def _is_ordered_subsequence(target_column, candidate_column):
    target_values = [_value_key(v) for v in target_column]
    candidate_values = [_value_key(v) for v in candidate_column]

    candidate_index = 0

    for target_value in target_values:
        if (
            candidate_index < len(candidate_values)
            and target_value == candidate_values[candidate_index]
        ):
            candidate_index += 1

    return candidate_index == len(candidate_values)


def _counts_are_subset(target_counts, candidate_counts):
    return all(
        count <= target_counts[value] for value, count in candidate_counts.items()
    )


def find_matching_columns(
    target_column,
    candidate_table,
    partial=False,
):
    unordered_match = None

    target_counts = None
    target_sorted = None

    for column in candidate_table.columns:
        candidate_column = candidate_table[column]

        if len(candidate_column) > len(target_column):
            continue

        if partial:
            ordered = _is_ordered_subsequence(
                target_column,
                candidate_column,
            )

            if ordered:
                return column, column

            if target_counts is None:
                target_counts = _column_counts(target_column)

            candidate_counts = _column_counts(candidate_column)

            unordered = _counts_are_subset(
                target_counts,
                candidate_counts,
            )

        else:
            ordered = len(candidate_column) == len(
                target_column
            ) and target_column.equals(candidate_column)

            if ordered:
                return column, column

            if target_sorted is None:
                target_sorted = target_column.sort_values(ignore_index=True)

            candidate_sorted = candidate_column.sort_values(ignore_index=True)

            unordered = target_sorted.equals(candidate_sorted)

        if unordered and unordered_match is None:
            unordered_match = column

        # Continue because a later column might match in order.

    return None, unordered_match


def table_match_modes(
    target_table,
    candidate_table,
    partial=False,
):
    """
    Return:
        (all_columns_match_in_order,
         all_columns_match_ignoring_order)
    """
    all_ordered = True

    for column in target_table.columns:
        ordered_match, unordered_match = find_matching_columns(
            target_table[column],
            candidate_table,
            partial=partial,
        )

        if ordered_match is None:
            all_ordered = False

        # If one required column has no match at all, the table fails.
        if unordered_match is None:
            return False, False

    return all_ordered, True


def classify_table_match(
    target_table,
    candidate_table,
    min_coverage=0.5,
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
