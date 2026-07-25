import re
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput

from .eval_models import BenchmarkOutput, CaseMetadata, NumberOutput

FRACTION_PATTERN = re.compile(r"([+-]?\d+)\s*/\s*([+-]?\d+)")
PERCENTAGE_PATTERN = re.compile(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*%")
NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+\.\d+|\d+)")


def extract_numbers(text: str) -> list[float]:
    """Extract fractions, percentages, decimals, and integers from text."""
    values: list[float] = []

    for numerator, denominator in FRACTION_PATTERN.findall(text):
        try:
            values.append(float(Fraction(int(numerator), int(denominator))))
        except ZeroDivisionError:
            pass

    # Remove fractions before extracting other numbers so "1/2" is not also
    # interpreted as the two integers 1 and 2.
    remaining_text = FRACTION_PATTERN.sub(" ", text)

    for percentage in PERCENTAGE_PATTERN.findall(remaining_text):
        values.append(float(percentage) / 100.0)

    # Likewise, remove percentages so "50%" produces 0.5, not both 0.5 and 50.
    remaining_text = PERCENTAGE_PATTERN.sub(" ", remaining_text)

    for number in NUMBER_PATTERN.findall(remaining_text):
        values.append(float(number))

    return values


@dataclass
class ValidateNumberOutput(Evaluator[str, BenchmarkOutput, CaseMetadata]):
    """Check that every expected number appears within the configured tolerance."""

    tolerance: float = 0.01

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

        actual_values = extract_numbers(cast(str, ctx.output))
        for number_output in number_outputs:
            for actual_value in actual_values:
                if abs(actual_value - number_output.value) <= self.tolerance:
                    break
            else:
                return False

        return True
