import re
from collections import Counter
from dataclasses import dataclass
from typing import cast

from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput

from ..eval_models import BenchmarkOutput, CaseMetadata, TextOutput

_MARKDOWN_PATTERN = re.compile(
    r"(?<!\w)(`|\*\*|\*|__|_)(?=\S)(.+?)(?<=\S)\1(?!\w)", re.DOTALL
)


def normalize_text(value: str) -> str:
    """Ignore case, whitespace, and basic Markdown without altering identifiers."""
    while True:
        unwrapped = _MARKDOWN_PATTERN.sub(r"\2", value)
        if unwrapped == value:
            break
        value = unwrapped
    return re.sub(r"\s+", " ", value.lower()).strip()


@dataclass
class ValidateTextOutput(Evaluator[str, BenchmarkOutput, CaseMetadata]):
    """Require distinct bounded matches for expected text within the response."""

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
        expected_texts = [
            normalize_text(value)
            for text_output in text_outputs
            for value in text_output.text
        ]
        if not all(expected_texts):
            return False

        # Prefer longer phrases at the same position. A single scan makes matches
        # non-overlapping; Counter preserves repeated expected occurrences.
        alternatives = "|".join(
            re.escape(value)
            for value in sorted(
                set(expected_texts), key=lambda value: (-len(value), value)
            )
        )
        matches = re.findall(rf"(?<!\w)(?:{alternatives})(?!\w)", actual_text)
        return not (Counter(expected_texts) - Counter(matches))
