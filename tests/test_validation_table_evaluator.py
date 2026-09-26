from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.validation.eval_models import TableOutput, TextOutput
from src.validation.evaluators.table import (
    ValidateTableIncluded,
    ValidateTableValues,
    classify_table_match,
)


@pytest.mark.parametrize(
    ("candidate", "match_type"),
    [
        ({"l1": [0.8], "l2": [0.9]}, "exact_match"),
        ({"l2": [0.9], "l1": [0.8]}, "exact_match"),
        ({"l1": [0.9], "l2": [0.8]}, "no_match"),
        ({"phase_one": [0.8], "phase_two": [0.9]}, "exact_match"),
        ({"phase_one": [0.9], "phase_two": [0.8]}, "no_match"),
        ({"l2": [0.9], "phase_one": [0.8]}, "exact_match"),
        ({"extra": [100], "l2": [0.9], "l1": [0.8]}, "exact_match"),
        ({"l1": [0.8]}, "no_match"),
    ],
)
def test_column_mapping_uses_names_then_positions(candidate, match_type) -> None:
    target = pd.DataFrame({"l1": [0.8], "l2": [0.9]})
    assert classify_table_match(target, pd.DataFrame(candidate)) == (match_type, 1.0)


def test_one_candidate_column_cannot_satisfy_two_expected_columns() -> None:
    target = pd.DataFrame({"a": [1, 2], "b": [1, 2]})
    candidate = pd.DataFrame({"a": [1, 2]})
    assert classify_table_match(target, candidate) == ("no_match", 1.0)


@pytest.mark.parametrize("duplicate_target", [False, True])
def test_duplicate_column_names_are_ambiguous(duplicate_target) -> None:
    target = pd.DataFrame([[1, 2]], columns=["a", "b"])
    candidate = target.copy()
    (target if duplicate_target else candidate).columns = ["a", "a"]
    assert classify_table_match(target, candidate) == ("no_match", 1.0)


@pytest.mark.parametrize(
    ("candidate", "match_type", "coverage"),
    [
        ({"device": ["a", "b"], "current": [10, 20]}, "exact_match", 1.0),
        ({"device": ["b", "a"], "current": [20, 10]}, "match_up_to_ordering", 1.0),
        ({"device": ["a", "b"], "current": [20, 10]}, "no_match", 1.0),
        ({"device": ["a", "a"], "current": [10, 10]}, "no_match", 1.0),
        ({"device": ["b"], "current": [20]}, "partial_exact_match", 0.5),
        ({"device": ["a"], "current": [20]}, "no_match", 0.5),
        ({"device": [], "current": []}, "no_match", 0.0),
        ({"device": ["a", "b", "b"], "current": [10, 20, 20]}, "no_match", 1.5),
    ],
)
def test_whole_rows_preserve_associations_and_multiplicity(
    candidate, match_type, coverage
) -> None:
    target = pd.DataFrame({"device": ["a", "b"], "current": [10, 20]})
    assert classify_table_match(target, pd.DataFrame(candidate)) == (
        match_type,
        coverage,
    )


def test_partial_unordered_match_uses_whole_rows_and_coverage_threshold() -> None:
    target = pd.DataFrame({"id": ["a", "b", "c"], "value": [10, 20, 30]})
    candidate = target.iloc[[2, 0]]
    assert classify_table_match(target, candidate) == (
        "partial_match_up_to_ordering",
        2 / 3,
    )
    assert classify_table_match(target, candidate, min_coverage=0.75) == (
        "no_match",
        2 / 3,
    )


@pytest.mark.parametrize(
    ("expected", "actual", "match_type"),
    [
        (1, 1.0, "exact_match"),
        (0.997, 0.997 + 1e-12, "exact_match"),
        (0.276, 0.2764, "exact_match"),
        (0.276, 0.2766, "no_match"),
        (0.0276, 0.02759, "exact_match"),
        (0.0276, 0.0275, "no_match"),
        (-0.0276, -0.02759, "exact_match"),
        (1930.1304, 1930.13, "exact_match"),
        (1930.1304, 1930.14, "no_match"),
        (0, 1e-9, "exact_match"),
        (0, 1e-8, "no_match"),
        (float("nan"), None, "exact_match"),
        (None, 0, "no_match"),
        (float("inf"), float("inf"), "no_match"),
        (True, 1, "no_match"),
        (True, True, "exact_match"),
        ("device_1", "device_10", "no_match"),
        ("Exact description.", "exact description.", "no_match"),
        ("1", 1, "no_match"),
    ],
)
def test_cell_comparison(expected, actual, match_type) -> None:
    assert classify_table_match(
        pd.DataFrame({"value": [expected]}), pd.DataFrame({"value": [actual]})
    ) == (match_type, 1.0)


@pytest.mark.parametrize(
    ("actual", "match_type"),
    [
        ("2025-03-10T09:13:03.172Z", "exact_match"),
        ("2025-03-10T09:13:03.172000+00:00", "exact_match"),
        ("2025-03-10T11:13:03.172+02:00", "exact_match"),
        ("2025-03-10T11:13:03.172Z", "no_match"),
        ("2025-03-10T09:13:03.172000001Z", "no_match"),
        ("2025-03-10T09:13:03Z", "no_match"),
        ("2025-99-10T09:13:03.172Z", "no_match"),
    ],
)
def test_iso_timestamps_compare_instants_without_losing_precision(
    actual, match_type
) -> None:
    target = pd.DataFrame({"time": ["2025-03-10 09:13:03.172"]})
    candidate = pd.DataFrame({"time": [actual]})
    assert classify_table_match(target, candidate) == (match_type, 1.0)


def test_date_and_midnight_timestamp_are_equivalent() -> None:
    target = pd.DataFrame({"date": ["2025-03-10"]})
    candidate = pd.DataFrame({"date": ["2025-03-10T00:00:00Z"]})
    assert classify_table_match(target, candidate) == ("exact_match", 1.0)


def test_overlapping_tolerances_do_not_make_row_matching_greedy() -> None:
    target = pd.DataFrame({"value": [1.0, 1.008]})
    # The first candidate fits either row, but the second only fits the first.
    candidate = pd.DataFrame({"value": [1.004, 0.996]})
    assert classify_table_match(target, candidate) == ("match_up_to_ordering", 1.0)


def test_missing_values_and_duplicate_rows_keep_their_multiplicity() -> None:
    target = pd.DataFrame({"value": [None, 1, None, 2]})
    assert classify_table_match(target, target.iloc[[2, 3, 0, 1]]) == (
        "match_up_to_ordering",
        1.0,
    )
    candidate = pd.DataFrame({"value": [None, 1, 1, 2]})
    assert classify_table_match(target, candidate) == ("no_match", 1.0)


@pytest.mark.parametrize(
    ("candidate", "match_type", "coverage"),
    [
        ({"a": [], "b": []}, "exact_match", 1.0),
        ({"alias_a": [], "alias_b": []}, "exact_match", 1.0),
        ({"a": []}, "no_match", None),
        ({"a": [1], "b": [2]}, "no_match", None),
    ],
)
def test_empty_tables_still_require_distinct_columns(
    candidate, match_type, coverage
) -> None:
    target = pd.DataFrame(columns=["a", "b"])
    assert classify_table_match(target, pd.DataFrame(candidate)) == (
        match_type,
        coverage,
    )


@pytest.mark.parametrize(
    ("row_indices", "ordered", "accepted", "match_type"),
    [
        ([0, 1, 2], True, True, "exact_match"),
        ([2, 1, 0], True, False, "match_up_to_ordering"),
        ([2, 1, 0], False, True, "match_up_to_ordering"),
        ([0, 2], False, False, "partial_exact_match"),
        ([2, 0], False, False, "partial_match_up_to_ordering"),
        ([0, 0, 2], False, False, "no_match"),
    ],
)
def test_csv_evaluator_respects_order_and_rejects_partial_tables(
    tmp_path, row_indices, ordered, accepted, match_type
) -> None:
    target = pd.DataFrame({"id": ["a", "b", "c"], "value": [10, 20, 30]})
    target.to_csv(tmp_path / "target.csv", index=False)
    target.iloc[row_indices].to_csv(tmp_path / "candidate.csv", index=False)
    ctx = SimpleNamespace(
        expected_output=[TableOutput(table_path="target.csv", ordered=ordered)],
        attributes={"csv_path": tmp_path / "candidate.csv"},
    )
    assert ValidateTableIncluded().evaluate(cast(Any, ctx)) is True
    assert ValidateTableValues(artifact_base_path=tmp_path).evaluate(
        cast(Any, ctx)
    ) == {
        "ValidateTableValues": accepted,
        "TableMatchType": match_type,
    }


def test_csv_evaluator_normalizes_values_and_honors_custom_tolerance(tmp_path) -> None:
    (tmp_path / "target.csv").write_text("time,value\n2025-03-10 09:13:03.172,0.276\n")
    (tmp_path / "candidate.csv").write_text(
        "time,value\n2025-03-10T11:13:03.172+02:00,0.2764\n"
    )
    ctx = SimpleNamespace(
        expected_output=[TableOutput(table_path="target.csv", ordered=True)],
        attributes={"csv_path": str(tmp_path / "candidate.csv")},
    )
    assert ValidateTableValues(artifact_base_path=tmp_path).evaluate(
        cast(Any, ctx)
    ) == {
        "ValidateTableValues": True,
        "TableMatchType": "exact_match",
    }
    assert ValidateTableValues(artifact_base_path=tmp_path, tolerance=0).evaluate(
        cast(Any, ctx)
    ) == {"ValidateTableValues": False, "TableMatchType": "no_match"}


def test_csv_evaluator_checks_every_expected_table(tmp_path) -> None:
    (tmp_path / "first.csv").write_text("value\n1\n")
    (tmp_path / "second.csv").write_text("value\n2\n")
    ctx = SimpleNamespace(
        expected_output=[
            TableOutput(table_path=str(tmp_path / "first.csv"), ordered=True),
            TableOutput(table_path=str(tmp_path / "second.csv"), ordered=True),
        ],
        attributes={"csv_path": tmp_path / "first.csv"},
    )
    assert ValidateTableValues().evaluate(cast(Any, ctx)) == {
        "ValidateTableValues": False,
        "TableMatchType": "no_match",
    }


@pytest.mark.parametrize("expected_output", [None, "text", [TextOutput(text=["text"])]])
def test_table_evaluators_skip_cases_without_tables(expected_output) -> None:
    ctx = SimpleNamespace(expected_output=expected_output, attributes={})
    assert ValidateTableIncluded().evaluate(cast(Any, ctx)) == {}
    assert ValidateTableValues().evaluate(cast(Any, ctx)) == {}


def test_missing_candidate_csv_fails() -> None:
    ctx = SimpleNamespace(
        expected_output=[TableOutput(table_path="unused.csv", ordered=False)],
        attributes={},
    )
    assert ValidateTableIncluded().evaluate(cast(Any, ctx)) is False
    assert ValidateTableValues().evaluate(cast(Any, ctx)) == {
        "ValidateTableValues": False,
        "TableMatchType": "no_match",
    }
