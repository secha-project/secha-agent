from dataclasses import dataclass
from pathlib import Path

from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput

from ..eval_models import BenchmarkOutput, CaseMetadata, VisualizationOutput


def _visualization_outputs(expected_output: BenchmarkOutput) -> list[VisualizationOutput]:
    if not isinstance(expected_output, list):
        return []
    return [
        item for item in expected_output if isinstance(item, VisualizationOutput)
    ]


@dataclass
class ValidateVisualizationIncluded(
    Evaluator[str, BenchmarkOutput, CaseMetadata]
):
    """Placeholder evaluator that checks visualization artifacts were produced."""

    def evaluate(
        self,
        ctx: EvaluatorContext[str, BenchmarkOutput, CaseMetadata],
    ) -> EvaluatorOutput:
        if not _visualization_outputs(ctx.expected_output):
            return {}

        spec_path = ctx.attributes.get("visualization_spec_path")
        image_path = ctx.attributes.get("visualization_image_path")
        return bool(
            isinstance(spec_path, (str, Path))
            and isinstance(image_path, (str, Path))
            and Path(spec_path).is_file()
            and Path(image_path).is_file()
        )
