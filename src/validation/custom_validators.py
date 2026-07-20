"""Custom evaluators for benchmark output types."""

from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput

from .eval_models import BenchmarkOutput, CaseMetadata, TextOutput


@dataclass
class ValidateTextOutput(Evaluator[str, BenchmarkOutput, CaseMetadata]):
    """Check that every expected text value appears in the agent response."""

    def evaluate(
        self, ctx: EvaluatorContext[str, BenchmarkOutput, CaseMetadata]
    ) -> EvaluatorOutput:
        expected_output = ctx.expected_output
        if isinstance(expected_output, TextOutput):
            text_outputs = [expected_output]
        elif isinstance(expected_output, list):
            text_outputs = [
                item for item in expected_output if isinstance(item, TextOutput)
            ]
        else:
            text_outputs = []

        if not text_outputs:
            return {}

        if not isinstance(ctx.output, str):
            return False

        return all(
            text in ctx.output
            for text_output in text_outputs
            for text in text_output.text
        )


CUSTOM_EVALUATOR_TYPES = (ValidateTextOutput,)
