import re
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput

from ..eval_models import BenchmarkOutput, CaseMetadata, NumberOutput
from ._numeric import numbers_match

FRACTION_PATTERN = re.compile(r"([+-]?\d+)\s*/\s*([+-]?\d+)")
NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
PERCENTAGE_PATTERN = re.compile(rf"({NUMBER_PATTERN.pattern})\s*%")


def extract_numbers(text: str) -> list[float]:
    """Extract fractions, percentages, decimals, and integers in textual order."""
    text = text.replace("−", "-")
    values: list[tuple[int, float]] = []

    for match in FRACTION_PATTERN.finditer(text):
        numerator, denominator = match.groups()
        try:
            values.append(
                (match.start(), float(Fraction(int(numerator), int(denominator))))
            )
        except ZeroDivisionError:
            pass

    # Mask fractions without shifting positions or counting their components again.
    remaining_text = FRACTION_PATTERN.sub(lambda match: " " * len(match[0]), text)

    for match in PERCENTAGE_PATTERN.finditer(remaining_text):
        values.append((match.start(), float(match[1]) / 100.0))

    # Likewise, remove percentages so "50%" produces 0.5, not both 0.5 and 50.
    remaining_text = PERCENTAGE_PATTERN.sub(
        lambda match: " " * len(match[0]), remaining_text
    )

    for match in NUMBER_PATTERN.finditer(remaining_text):
        values.append((match.start(), float(match[0])))

    return [value for _, value in sorted(values)]


@dataclass
class ValidateNumberOutput(Evaluator[str, BenchmarkOutput, CaseMetadata]):
    """Match numbers once and in order, tightening tolerance for small values."""

    tolerance: float = 0.005

    def evaluate(
        self, ctx: EvaluatorContext[str, BenchmarkOutput, CaseMetadata]
    ) -> EvaluatorOutput:
        expected_output = ctx.expected_output
        if not isinstance(expected_output, list):
            return {}

        number_outputs = [
            item for item in expected_output if isinstance(item, NumberOutput)
        ]

        if not number_outputs:
            return {}

        actual_values = iter(extract_numbers(cast(str, ctx.output)))
        for expected_value in (
            value for number_output in number_outputs for value in number_output.value
        ):
            for actual_value in actual_values:
                if numbers_match(expected_value, actual_value, self.tolerance):
                    break
            else:
                return False

        return True
