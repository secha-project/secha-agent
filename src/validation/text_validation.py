import re
from dataclasses import dataclass
from typing import cast

from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput

from .eval_models import BenchmarkOutput, CaseMetadata, TextOutput


def normalize_text(value: str) -> str:
    """Normalize case and whitespace while preserving punctuation."""
    return re.sub(r"\s+", " ", value.lower()).strip()


@dataclass
class ValidateTextOutput(Evaluator[str, BenchmarkOutput, CaseMetadata]):
    """Check that every expected text value appears in the agent response."""

    def evaluate(
        self, ctx: EvaluatorContext[str, BenchmarkOutput, CaseMetadata]
    ) -> EvaluatorOutput:
        expected_output = ctx.expected_output
        if not isinstance(expected_output, list):
            return {}

        text_outputs = [
            item for item in expected_output if isinstance(item, TextOutput)
        ]

        if not text_outputs:
            return {}

        actual_text = normalize_text(cast(str, ctx.output))
        for text_output in text_outputs:
            for expected_text in text_output.text:
                if normalize_text(expected_text) not in actual_text:
                    return False

        return True