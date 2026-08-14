from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, BinaryContent, ModelSettings
from pydantic_ai.models import KnownModelName, Model
from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
    EvaluatorOutput,
)

from ..eval_models import BenchmarkOutput, CaseMetadata, VisualizationOutput

_RUBRIC = (
    Path(__file__).with_name("visualization_rubric.md").read_text(encoding="utf-8")
)
_DEFAULT_JUDGE_MODEL = "openai:gpt-5.4-mini"

RubricScore = Annotated[int, Field(ge=0, le=2)]


class VisualizationEvaluation(BaseModel):
    """Structured assessment returned by the visualization judge."""

    model_config = ConfigDict(extra="forbid")

    visualization_type: RubricScore
    data_encoding: RubricScore
    data_transformation: RubricScore
    aesthetics: RubricScore
    is_blank: Literal[0, 1]
    rationale: Annotated[str, Field(min_length=1)]


def _visualization_outputs(
    expected_output: BenchmarkOutput | None,
) -> list[VisualizationOutput]:
    if not isinstance(expected_output, list):
        return []
    return [item for item in expected_output if isinstance(item, VisualizationOutput)]


@dataclass
class ValidateVisualizationIncluded(Evaluator[str, BenchmarkOutput, CaseMetadata]):
    """Check that the agent produced visualization spec and image artifacts."""

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


@dataclass
class ValidateVisualizationQuality(Evaluator[str, BenchmarkOutput, CaseMetadata]):
    """Use a multimodal LLM to compare a generated plot with its reference."""

    artifact_base_path: Path = Path()
    model: Model | KnownModelName | str = _DEFAULT_JUDGE_MODEL

    async def evaluate(
        self,
        ctx: EvaluatorContext[str, BenchmarkOutput, CaseMetadata],
    ) -> EvaluatorOutput:
        expected_visualizations = _visualization_outputs(ctx.expected_output)
        if not expected_visualizations:
            return {}

        generated_path_value = ctx.attributes.get("visualization_image_path")
        if not isinstance(generated_path_value, (str, Path)):
            return {}

        generated_path = Path(generated_path_value)
        if not generated_path.is_file():
            return {}

        # A run currently produces one visualization, so compare it with the
        # first reference visualization declared by the benchmark case.
        reference_path = Path(expected_visualizations[0].image_path)
        if not reference_path.is_absolute():
            reference_path = self.artifact_base_path / reference_path

        judge = Agent(
            self.model,
            output_type=VisualizationEvaluation,
            instructions=_RUBRIC,
            model_settings=ModelSettings(temperature=0.0),
        )
        result = await judge.run(
            [
                "Generated plot (first image):",
                BinaryContent.from_path(generated_path),
                "Ground-truth plot (second image):",
                BinaryContent.from_path(reference_path),
            ]
        )
        assessment = result.output

        # Pydantic Evals treats floats as scores and expects them in [0, 1].
        # Retain the rubric's integer scale in the structured judge output and
        # normalize only at the evaluator boundary.
        rationale = assessment.rationale
        return {
            "visualization_type": EvaluationReason(
                value=assessment.visualization_type / 2,
                reason=rationale,
            ),
            "data_encoding": EvaluationReason(
                value=assessment.data_encoding / 2,
                reason=rationale,
            ),
            "data_transformation": EvaluationReason(
                value=assessment.data_transformation / 2,
                reason=rationale,
            ),
            "aesthetics": EvaluationReason(
                value=assessment.aesthetics / 2,
                reason=rationale,
            ),
            "visualization_not_blank": EvaluationReason(
                value=assessment.is_blank == 0,
                reason=rationale,
            ),
        }
