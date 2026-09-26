from types import SimpleNamespace
from typing import Any, cast

import pytest

from src.validation.eval_models import NumberOutput, TextOutput
from src.validation.evaluators.number import ValidateNumberOutput, extract_numbers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2, 1/2, 25%, -3.5, .75%, 4/2, +7", [2, 0.5, 0.25, -3.5, 0.0075, 2, 7]),
        ("50%, 1/2, 0.5", [0.5, 0.5, 0.5]),
        ("4, 1/0, 25%, 2/4", [4, 0.25, 0.5]),
        (".0276, −.3, 2.76e-2, -3E+2, 1.e2", [0.0276, -0.3, 0.0276, -300, 100]),
        ("−1/2, 2e1%, .75%, 3e−2", [-0.5, 0.2, 0.0075, 0.03]),
        ("125,314,697", [125, 314, 697]),
        ("No numeric result", []),
    ],
)
def test_extract_numbers_preserves_occurrence_order(text, expected) -> None:
    assert extract_numbers(text) == expected


@pytest.mark.parametrize(
    ("expected", "output", "accepted"),
    [
        ([19, 19, 9], "19, 19, 9", True),
        ([19, 19, 9], "19, 0, 9", False),
        ([19, 19, 9], "19, 9", False),
        ([19, 19, 9], "19, 9, 19", False),
        ([125, 314, 697], "125, 314, 697", True),
        ([125, 314, 697], "697, 314, 125", False),
        ([-0.062856, 0.048392], "0.048392, -0.062856", False),
        ([2, 0.5, 0.25], "2, 1/2, 25%", True),
        ([2, 0.5, 0.25], "25%, 1/2, 2", False),
        ([0.5, 0.5, 0.5], "1/2, 50%, 0.5", True),
        ([0.5, 0.5], "1/2", False),
        ([0.5, 0.5], "50%", False),
        ([1, 2], "1/2", False),
        ([0.5, 50], "50%", False),
        ([5.25, 9.75], "Meter 156: first 5.25; at 12:30, second 9.75", True),
        ([1], "1.004", True),
        ([1], "1.006", False),
        ([1, 1.005], "1.005", False),
        ([1, 1.005], "1.005, 1.005", True),
        ([0.0276], ".0276", True),
        ([0.0276], "2.76e-2", True),
        ([-0.062856, 0.048392], "−0.062856, 4.8392e-2", True),
        ([0.0276, 0.0276], "2.76e-2", False),
        ([2.76, -2], "2.76e-2", False),
        ([0.2, 20], "2e1%", False),
        ([1], "", False),
    ],
)
def test_number_matching_requires_distinct_ordered_occurrences(
    expected, output, accepted
) -> None:
    ctx = SimpleNamespace(expected_output=[NumberOutput(value=expected)], output=output)
    assert ValidateNumberOutput().evaluate(cast(Any, ctx)) is accepted


@pytest.mark.parametrize(
    ("expected", "output", "accepted"),
    [
        (0.0276, "0.02", False),
        (0.0276, "0.02759119347", True),
        (0.0276, "0.0275", False),
        (0.0276, "0.0277", False),
        (0.276, "0.2764", True),
        (0.276, "0.2766", False),
        (-0.276, "-0.2764", True),
        (-0.276, "-0.2766", False),
        (0.00276, "0.002764", True),
        (0.00276, "0.002766", False),
        (16.56679385556739, "16.57", True),
        (186.51617000000002, "186.52", True),
        (1930.1304, "1930.13", True),
        (1930.1304, "1930.14", False),
        (1000000, "1000000.004", True),
        (1000000, "1000000.006", False),
        (0, "0", True),
        (0, "1e-9", True),
        (0, "1e-8", False),
        (float("nan"), "0", False),
        (float("inf"), "1e999", False),
    ],
)
def test_tolerance_tightens_by_magnitude_and_is_capped(
    expected, output, accepted
) -> None:
    ctx = SimpleNamespace(
        expected_output=[NumberOutput(value=[expected])], output=output
    )
    assert ValidateNumberOutput().evaluate(cast(Any, ctx)) is accepted


@pytest.mark.parametrize(
    ("output", "accepted"),
    [("19, 19, 9", True), ("19, 9", False), ("9, 19, 19", False)],
)
def test_number_output_groups_share_the_same_ordered_matches(output, accepted) -> None:
    ctx = SimpleNamespace(
        expected_output=[
            NumberOutput(value=[19]),
            TextOutput(text=["description"]),
            NumberOutput(value=[19, 9]),
        ],
        output=output,
    )
    assert ValidateNumberOutput().evaluate(cast(Any, ctx)) is accepted


@pytest.mark.parametrize("expected_output", [None, "text", [TextOutput(text=["text"])]])
def test_number_evaluator_skips_outputs_without_numeric_expectations(
    expected_output,
) -> None:
    ctx = SimpleNamespace(expected_output=expected_output, output="text")
    assert ValidateNumberOutput().evaluate(cast(Any, ctx)) == {}
