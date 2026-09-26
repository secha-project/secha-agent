from types import SimpleNamespace
from typing import Any, cast

import pytest

from src.validation.eval_models import (
    NumberOutput,
    TableOutput,
    TextOutput,
    VisualizationOutput,
)
from src.validation.evaluators.text import ValidateTextOutput, normalize_text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  EV-charging\n\tstation  ", "ev-charging station"),
        ("EV-charging **station**", "ev-charging station"),
        ("**EV-charging station**", "ev-charging station"),
        ("*EV-charging station*", "ev-charging station"),
        ("__EV-charging station__", "ev-charging station"),
        ("_EV-charging station_", "ev-charging station"),
        ("**`device_21`**", "device_21"),
        ("`power_quality_measurements_procem`", "power_quality_measurements_procem"),
        ("power_quality_measurements_procem", "power_quality_measurements_procem"),
        ("_device_21_", "device_21"),
        ("kWh/h", "kwh/h"),
        ("k*Wh", "k*wh"),
        ("2025-02-28 to 2025-03-31", "2025-02-28 to 2025-03-31"),
        ("19:02", "19:02"),
    ],
)
def test_normalization_preserves_identifiers_and_meaningful_punctuation(
    text, expected
) -> None:
    assert normalize_text(text) == expected


@pytest.mark.parametrize(
    ("expected", "output", "accepted"),
    [
        ("EV-charging station", "EV-charging station", True),
        ("EV-charging station", " EV-CHARGING\nstation ", True),
        ("EV-charging station", "EV-charging **station**", True),
        (
            "EV-charging station",
            "**EV-charging station**\n\nThe metadata describes eight AC charging spots.",
            True,
        ),
        ("EV-charging station", "individual EV-charger", False),
        ("EV-charging station", "charging station", False),
        ("kWh", "`kWh`", True),
        ("kWh", "The unit is kWh", True),
        ("2025-02-28 to 2025-03-31", "2025-02-28 to 2025-03-31", True),
        (
            "2025-02-28 to 2025-03-31",
            (
                "Earliest: 2025-02-28 21:59:59.774 UTC. "
                "Latest: 2025-03-31 23:59:59.788 UTC.\n\n"
                "**Answer: 2025-02-28 to 2025-03-31**"
            ),
            True,
        ),
        ("device_21", "device_210", False),
        ("device_21", "", False),
        (" ", "", False),
    ],
)
def test_single_text_answer_allows_surrounding_prose(
    expected, output, accepted
) -> None:
    ctx = SimpleNamespace(expected_output=[TextOutput(text=[expected])], output=output)
    assert ValidateTextOutput().evaluate(cast(Any, ctx)) is accepted


@pytest.mark.parametrize(
    ("expected", "output"),
    [
        ("EV-charging station", "It is not an EV-charging station."),
        ("EV-charging station", "individual EV-charger or EV-charging station"),
        ("kWh", "kWh/h"),
        ("2025-02-28 to 2025-03-31", "Not 2025-02-28 to 2025-03-31"),
    ],
)
def test_presence_matching_does_not_interpret_surrounding_context(
    expected, output
) -> None:
    ctx = SimpleNamespace(expected_output=[TextOutput(text=[expected])], output=output)
    assert ValidateTextOutput().evaluate(cast(Any, ctx)) is True


@pytest.mark.parametrize(
    ("expected", "output", "accepted"),
    [
        (["device_a", "device_b"], "device_a, device_b", True),
        (["device_a", "device_b"], "Devices: `device_b` and **device_a**.", True),
        (["device_a", "device_b"], "device_a_backup, device_b", False),
        (["device_a", "device_b"], "other_device_a, device_b", False),
        (["device_a", "device_b"], "device_a, device_bc", False),
        (["device_a", "device_b"], "device_a, device_a", False),
        (["device_a", "device_a"], "device_a", False),
        (["device_a", "device_a"], "device_a, device_a", True),
        (["DEVICE_A", "device_a"], "device_a", False),
        (["charging", "charging station"], "charging station", False),
        (["charging", "charging station"], "charging station, charging", True),
        (["charging station", "charging"], "charging, charging station", True),
        (["station", "station"], "station station", True),
        (["station", "station"], "station", False),
        (["a+b", "a.b"], "a+b, a.b", True),
        (["a+b", "a.b"], "aaab, axb", False),
    ],
)
def test_lists_use_distinct_nonoverlapping_matches_in_any_order(
    expected, output, accepted
) -> None:
    ctx = SimpleNamespace(expected_output=[TextOutput(text=expected)], output=output)
    assert ValidateTextOutput().evaluate(cast(Any, ctx)) is accepted


@pytest.mark.parametrize(
    ("expected", "output", "accepted"),
    [
        (["device_21", "19:02"], "device_21, 49.881123 Hz, 19:02 UTC", True),
        (["device_21", "19:02"], "device_210, 49.881123 Hz, 19:02 UTC", False),
        (["device_21", "19:02"], "device_21, 49.881123 Hz, 119:02 UTC", False),
        (["device_21", "19:02"], "device_21, 49.881123 Hz, 19:020 UTC", False),
        (["device_21"], "device_21é, 49.881123 Hz", False),
        (
            ["power_quality_measurements_procem"],
            "power_quality_measurements_procem, 13",
            True,
        ),
        (
            ["power_quality_measurements_procem"],
            "power_quality_measurements_procem_archive, 13",
            False,
        ),
        (["device_21"], "device21, 49.881123 Hz", False),
    ],
)
def test_mixed_answers_use_bounded_matches(expected, output, accepted) -> None:
    ctx = SimpleNamespace(
        expected_output=[TextOutput(text=expected), NumberOutput(value=[13])],
        output=output,
    )
    assert ValidateTextOutput().evaluate(cast(Any, ctx)) is accepted


@pytest.mark.parametrize(
    "other_output",
    [
        NumberOutput(value=[13]),
        TableOutput(table_path="unused.csv", ordered=True),
        VisualizationOutput(image_path="unused.png"),
    ],
)
def test_single_text_with_other_components_allows_surrounding_prose(
    other_output,
) -> None:
    ctx = SimpleNamespace(
        expected_output=[TextOutput(text=["device_a"]), other_output],
        output="Results for device_a: 13.",
    )
    assert ValidateTextOutput().evaluate(cast(Any, ctx)) is True


@pytest.mark.parametrize(
    ("output", "accepted"),
    [("device_a", False), ("device_a, device_a", True)],
)
def test_separate_text_groups_cannot_reuse_one_match(output, accepted) -> None:
    ctx = SimpleNamespace(
        expected_output=[TextOutput(text=["device_a"]), TextOutput(text=["device_a"])],
        output=output,
    )
    assert ValidateTextOutput().evaluate(cast(Any, ctx)) is accepted


@pytest.mark.parametrize("expected_output", [None, "text", [NumberOutput(value=[13])]])
def test_evaluator_skips_cases_without_expected_text(expected_output) -> None:
    ctx = SimpleNamespace(expected_output=expected_output, output="text")
    assert ValidateTextOutput().evaluate(cast(Any, ctx)) == {}
