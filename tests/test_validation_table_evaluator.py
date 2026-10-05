import json
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.validation.eval_models import CaseMetadata, TableOutput, TextOutput
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


def test_explicit_field_aliases_match_reordered_columns_with_extra_data() -> None:
    target = pd.DataFrame(
        {
            "device_name": ["meter_a", "meter_b"],
            "timestamp": ["2025-03-01 00:00:00", "2025-03-01 00:01:00"],
            "ul1v": [230, 231],
            "ul2v": [232, 233],
            "freq": [50, 49.9],
        }
    )
    candidate = pd.DataFrame(
        {
            "timestamp_utc": ["2025-03-01T00:00Z", "2025-03-01T00:01Z"],
            "extra": [999, 999],
            "supply_voltage_l2_v": [232, 233],
            "frequency_hz": [50, 49.9],
            "device_identifier": ["meter_a", "meter_b"],
            "phase_l1_voltage_v": [230, 231],
        }
    )
    assert classify_table_match(target, candidate) == ("exact_match", 1.0)


@pytest.mark.parametrize(
    "time_name,power_name",
    [
        ("measurement_time_utc", "power_w"),
        ("timestamp", "measurement_power_w"),
    ],
)
def test_required_measurements_match_with_device_and_window_details(
    time_name, power_name
) -> None:
    target = pd.DataFrame({"timestamp": ["2025-03-01 00:00:00.172"], "pw": [25.118034]})
    candidate = pd.DataFrame(
        {
            "device_name": ["KampusareenaEVCharging"],
            time_name: ["2025-03-01T00:00:00.172Z"],
            power_name: [25.118034],
            "window_min_w": [25.118034],
            "window_max_w": [28.537029],
        }
    )
    assert classify_table_match(target, candidate) == ("exact_match", 1.0)


def test_alias_mapping_does_not_search_values_to_hide_swapped_phases() -> None:
    target = pd.DataFrame({"ul1v": [230], "ul2v": [240]})
    candidate = pd.DataFrame({"voltage_l2_v": [230], "voltage_l1_v": [240]})
    assert classify_table_match(target, candidate) == ("no_match", 1.0)


def test_ambiguous_duplicate_field_aliases_are_rejected() -> None:
    target = pd.DataFrame({"pw": [10]})
    candidate = pd.DataFrame({"power_w": [10], "active_power_w": [11]})
    assert classify_table_match(target, candidate) == ("no_match", 1.0)


def test_field_aliases_do_not_convert_units_or_relax_numeric_precision() -> None:
    target = pd.DataFrame({"pw": [29.174286]})
    candidate = pd.DataFrame({"power_kw": [0.029174286]})
    assert classify_table_match(target, candidate) == ("no_match", 1.0)
    assert classify_table_match(
        pd.DataFrame({"pw": [0.027344229]}),
        pd.DataFrame({"power_w": [0.027]}),
    ) == ("no_match", 1.0)


@pytest.mark.parametrize(
    "actual",
    [
        "2025-03-10 09:13 UTC",
        "2025-03-10T09:13Z",
        "2025-03-10T11:13+02:00",
        "2025-03-10 09:13:00 UTC",
    ],
)
def test_equivalent_minute_and_utc_suffix_timestamps(actual) -> None:
    target = pd.DataFrame({"timestamp": ["2025-03-10 09:13:00"]})
    assert classify_table_match(target, pd.DataFrame({"timestamp_utc": [actual]})) == (
        "exact_match",
        1.0,
    )


def test_minute_timestamp_format_does_not_erase_reference_milliseconds() -> None:
    target = pd.DataFrame({"timestamp": ["2025-03-10 09:13:00.172"]})
    candidate = pd.DataFrame({"timestamp_utc": ["2025-03-10 09:13 UTC"]})
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


@pytest.mark.parametrize(
    "expected,actual",
    [
        (0.4033333333333333, 0.40),
        (0.027344229275521877, 0.027),
        (27.288971018055552, 27.3),
        (0.006794343975, 0.0068),
        (-0.0245, -0.025),
    ],
)
def test_correct_rounding_accepts_the_benchmark_examples(expected, actual):
    assert classify_table_match(
        pd.DataFrame({"value": [expected]}),
        pd.DataFrame({"value": [actual]}),
        minimum_significant_digits=2,
    ) == ("exact_match", 1.0)


@pytest.mark.parametrize(
    "expected,actual",
    [
        (0.0276, 0.02),
        (3.9095, 3.8726),
        (0.9994531, 0.999990),
        (0.0366666667, 0.04),
        (0.00123, 0),
        (314, 310),
    ],
)
def test_rounding_does_not_accept_nearby_values_coarse_results_or_counts(
    expected, actual
):
    assert classify_table_match(
        pd.DataFrame({"value": [expected]}),
        pd.DataFrame({"value": [actual]}),
        minimum_significant_digits=2,
    ) == ("no_match", 1.0)


def test_rounding_requires_consistent_precision_across_the_column():
    reference = pd.DataFrame({"value": [0.027344, 0.043211]})
    assert classify_table_match(
        reference,
        pd.DataFrame({"value": [0.027, 0.0432]}),
        minimum_significant_digits=2,
    ) == ("no_match", 1.0)
    assert classify_table_match(
        reference,
        pd.DataFrame({"value": [0.027, 0.043]}),
        minimum_significant_digits=2,
    ) == ("exact_match", 1.0)


@pytest.mark.parametrize("significant_figures", [False, True])
def test_rounding_supports_fixed_decimal_places_or_significant_figures(
    significant_figures,
):
    reference = pd.DataFrame({"value": [0.027344, 12.3456]})
    actual = [0.0273, 12.3] if significant_figures else [0.027, 12.346]
    assert classify_table_match(
        reference,
        pd.DataFrame({"value": actual}),
        minimum_significant_digits=2,
    ) == ("exact_match", 1.0)


def test_rounding_does_not_discard_row_associations_order_or_multiplicity():
    reference = pd.DataFrame({"id": ["a", "b"], "value": [0.027344, 0.043211]})
    rounded = pd.DataFrame({"id": ["a", "b"], "value": [0.027, 0.043]})
    assert classify_table_match(
        reference, rounded.iloc[::-1], minimum_significant_digits=2
    ) == ("match_up_to_ordering", 1.0)
    for candidate in (
        pd.DataFrame({"id": ["a", "b"], "value": [0.043, 0.027]}),
        rounded.iloc[[0, 0]],
    ):
        assert classify_table_match(
            reference, candidate, minimum_significant_digits=2
        ) == ("no_match", 1.0)
    assert (
        classify_table_match(reference, rounded.iloc[:1], minimum_significant_digits=2)[
            0
        ]
        == "no_match"
    )


def test_rounded_values_compare_exactly_without_a_second_tolerance():
    reference = pd.DataFrame({"id": ["a", "b"], "value": [0.027344, 0.027444]})
    swapped = pd.DataFrame({"id": ["a", "b"], "value": [0.0274, 0.0273]})
    assert classify_table_match(reference, swapped, minimum_significant_digits=2) == (
        "no_match",
        1.0,
    )


@pytest.mark.parametrize(
    "inputs,accepted",
    [
        ("Show average consumption in kW.", True),
        ("Return the value to four decimal places.", False),
        ("Round to two decimals.", False),
        ("Return a six-decimal correlation.", False),
        ("Return two significant figures.", False),
        ("Show unrounded values.", False),
        ("Preserve full precision.", False),
        ("Do not round the values.", False),
    ],
)
def test_csv_rounding_respects_explicit_precision_in_the_question(
    tmp_path, inputs, accepted
):
    (tmp_path / "target.csv").write_text("value\n0.027344229275521877\n")
    (tmp_path / "candidate.csv").write_text("value\n0.027\n")
    ctx = SimpleNamespace(
        inputs=inputs,
        expected_output=[TableOutput(table_path="target.csv", ordered=True)],
        attributes={"csv_path": str(tmp_path / "candidate.csv")},
    )
    assert (
        ValidateTableValues(artifact_base_path=tmp_path).evaluate(cast(Any, ctx))[
            "ValidateTableValues"
        ]
        is accepted
    )
    assert (
        ValidateTableValues(artifact_base_path=tmp_path, tolerance=0).evaluate(
            cast(Any, ctx)
        )["ValidateTableValues"]
        is False
    )


@pytest.fixture
def charging_start_table_context(tmp_path):
    (tmp_path / "target.csv").write_text(
        "charging_start,(pw / 1000),rolling_avg,(power_change / 1000)\n"
        "2025-03-01 01:19:20.615,0.029174286,0.027048353905,0.002125932095\n"
    )
    return SimpleNamespace(
        inputs="Provide the measured value, rolling average, and increase in kW.",
        metadata=CaseMetadata(
            case_id="d5884b41-e3f7-4ef5-bb2e-5c445e9743e1",
            category="semantic integration",
        ),
        expected_output=[TableOutput(table_path="target.csv", ordered=True)],
        attributes={"csv_path": str(tmp_path / "candidate.csv")},
    )


@pytest.mark.parametrize(
    "header,values,accepted",
    [
        (
            "measured_value_w,rolling_average_w,increase_kw",
            "29.174286,27.048353905,0.002125932095",
            True,
        ),
        ("measured_value_W,rolling_avg_W,increase_kW", "29.174,27.048,0.002126", True),
        (
            "measured_w,rolling_average_w,increase_kw",
            "29.174286,27.048353905,0.002125932095",
            True,
        ),
        (
            "measured_value_kw,rolling_average_kw,increase_kw",
            "0.029174286,0.027048353905,0.002125932095",
            True,
        ),
        ("measured_value,rolling_average,increase_kw", "29.174,27.048,0.002126", False),
        (
            "measured_value_kw,rolling_average_kw,increase_kw",
            "29.174,27.048,0.002126",
            False,
        ),
        ("measured_value_w,rolling_average_w,increase_w", "29.174,27.048,2.126", False),
        ("measured_value_w,rolling_average_w,increase_kw", "31,27.048,0.002126", False),
        ("measured_value_w,rolling_average_w,increase_kw", "29.174,26,0.002126", False),
        (
            "measured_value_w,rolling_average_w,increase_kw",
            "29.174,27.048,0.0022",
            False,
        ),
    ],
)
def test_charging_start_accepts_explicit_equivalent_units_only(
    tmp_path, charging_start_table_context, header, values, accepted
):
    (tmp_path / "candidate.csv").write_text(
        f"potential_charging_start_utc,{header}\n2025-03-01 01:19:20.615,{values}\n"
    )
    result = ValidateTableValues(artifact_base_path=tmp_path).evaluate(
        cast(Any, charging_start_table_context)
    )
    assert result["ValidateTableValues"] is accepted


@pytest.mark.parametrize("different_case", [False, True])
def test_unit_compatibility_keeps_case_scope_and_exact_event_timestamps(
    tmp_path, charging_start_table_context, different_case
):
    timestamp = "2025-03-01 01:19:20.615" if different_case else "2025-03-01 01:19:20"
    (tmp_path / "candidate.csv").write_text(
        "potential_charging_start_utc,measured_value_w,rolling_average_w,increase_kw\n"
        f"{timestamp},29.174286,27.048353905,0.002125932095\n"
    )
    if different_case:
        charging_start_table_context.metadata = CaseMetadata(
            case_id="00000000-0000-0000-0000-000000000001",
            category="semantic integration",
        )
    assert (
        ValidateTableValues(artifact_base_path=tmp_path).evaluate(
            cast(Any, charging_start_table_context)
        )["ValidateTableValues"]
        is False
    )


@pytest.fixture
def voltage_thd_table_context(tmp_path):
    references = tmp_path / "artifacts" / "task_55"
    references.mkdir(parents=True)
    header = "phase,mean_voltage_harmonic_distortion_v\n"
    (references / "table.csv").write_text(header + "L1,3.9095\nL2,3.5781\nL3,3.7861\n")
    (references / "table-reported.csv").write_text(
        header + "L1,3.8726\nL2,3.5395\nL3,3.7543\n"
    )
    (references / "table.alternatives.json").write_text(
        json.dumps({"table_paths": ["table-reported.csv"]})
    )
    return SimpleNamespace(
        inputs="Return the phase and mean magnitude in volts, rounded to four decimal places and ordered by phase.",
        metadata=CaseMetadata(
            case_id="968808f7-503e-4c5a-a859-597dfbac2d56",
            category="domain knowledge",
        ),
        expected_output=[
            TableOutput(table_path="artifacts/task_55/table.csv", ordered=True)
        ],
        attributes={"csv_path": str(tmp_path / "candidate.csv")},
    )


@pytest.mark.parametrize(
    "rows,accepted",
    [
        ("L1,3.9095\nL2,3.5781\nL3,3.7861\n", True),
        ("L1,3.8726\nL2,3.5395\nL3,3.7543\n", True),
        ("L1,3.8733\nL2,3.5402\nL3,3.7550\n", True),
        ("L1,3.9293\nL2,3.5961\nL3,3.8071\n", False),
        ("L1,3.9095\nL2,3.5395\nL3,3.7543\n", False),
        ("L1,3.5395\nL2,3.8726\nL3,3.7543\n", False),
        ("L1,3.8726\nL1,3.8726\nL3,3.7543\n", False),
        ("L1,3.8726\nL2,3.5395\n", False),
        ("L3,3.7543\nL2,3.5395\nL1,3.8726\n", False),
        ("L1,3.9\nL2,3.5\nL3,3.8\n", False),
        ("L1,1.6\nL2,1.5\nL3,1.6\n", False),
        ("L1,3.8726\nL2,3.5395\nL3,3.7543\nL4,3.8726\n", False),
    ],
)
def test_voltage_thd_accepts_either_complete_reference_only(
    tmp_path, voltage_thd_table_context, rows, accepted
):
    (tmp_path / "candidate.csv").write_text("phase,mean_thd_v\n" + rows)
    result = ValidateTableValues(artifact_base_path=tmp_path).evaluate(
        cast(Any, voltage_thd_table_context)
    )
    assert result["ValidateTableValues"] is accepted


def test_alternative_reference_preserves_unordered_table_matching(
    tmp_path, voltage_thd_table_context
):
    (tmp_path / "candidate.csv").write_text(
        "phase,mean_thd_v\nL3,3.7543\nL2,3.5395\nL1,3.8726\n"
    )
    voltage_thd_table_context.expected_output[0].ordered = False
    assert ValidateTableValues(artifact_base_path=tmp_path).evaluate(
        cast(Any, voltage_thd_table_context)
    ) == {"ValidateTableValues": True, "TableMatchType": "match_up_to_ordering"}


def test_alternative_reference_is_opted_in_by_the_artifact_only(
    tmp_path, voltage_thd_table_context
):
    references = tmp_path / "artifacts" / "task_55"
    (references / "table.alternatives.json").unlink()
    (tmp_path / "candidate.csv").write_text(
        "phase,mean_thd_v\nL1,3.8726\nL2,3.5395\nL3,3.7543\n"
    )
    assert (
        ValidateTableValues(artifact_base_path=tmp_path).evaluate(
            cast(Any, voltage_thd_table_context)
        )["ValidateTableValues"]
        is False
    )


def test_alternative_reference_keeps_the_configured_numeric_tolerance(
    tmp_path, voltage_thd_table_context
):
    (tmp_path / "candidate.csv").write_text(
        "phase,mean_thd_v\nL1,3.8729\nL2,3.5399\nL3,3.7546\n"
    )
    assert (
        ValidateTableValues(artifact_base_path=tmp_path, tolerance=0).evaluate(
            cast(Any, voltage_thd_table_context)
        )["ValidateTableValues"]
        is False
    )
