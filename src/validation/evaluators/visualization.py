from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool
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


class VisualizationEvaluation(BaseModel):
    """Structured assessment returned by the visualization judge."""

    model_config = ConfigDict(extra="forbid")

    data_correct: StrictBool
    encoding_correct: StrictBool
    type_correct: StrictBool
    aesthetics_correct: StrictBool
    rationale: Annotated[str, Field(min_length=1)]


def _visualization_outputs(
    expected_output: BenchmarkOutput | None,
) -> list[VisualizationOutput]:
    if not isinstance(expected_output, list):
        return []
    return [item for item in expected_output if isinstance(item, VisualizationOutput)]


def _load_png(path: Path, *, description: str) -> BinaryContent:
    """Load a PNG produced by the visualization artifact pipeline."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")

    data = path.read_bytes()
    if not data:
        raise ValueError(f"{description} is empty: {path}")

    return BinaryContent(
        data=data,
        media_type="image/png",
        vendor_metadata={"detail": "high"},
    )


def _failed_quality(reason: str) -> EvaluatorOutput:
    return {
        name: EvaluationReason(value=False, reason=reason)
        for name in (
            "data_correct",
            "encoding_correct",
            "type_correct",
            "aesthetics_correct",
        )
    }


@dataclass
class ValidateVisualizationIncluded(Evaluator[str, BenchmarkOutput, CaseMetadata]):
    """Check that the agent produced a valid visualization image."""

    def evaluate(
        self,
        ctx: EvaluatorContext[str, BenchmarkOutput, CaseMetadata],
    ) -> EvaluatorOutput:
        if not _visualization_outputs(ctx.expected_output):
            return {}

        image_path = ctx.attributes.get("visualization_image_path")
        if not isinstance(image_path, (str, Path)):
            return {"graph_included": False}

        try:
            _load_png(Path(image_path), description="Generated visualization image")
        except (OSError, ValueError):
            return {"graph_included": False}
        return {"graph_included": True}


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
            return _failed_quality("No generated visualization image was produced.")

        generated_path = Path(generated_path_value)
        try:
            generated_image = _load_png(
                generated_path,
                description="Generated visualization image",
            )
        except (OSError, ValueError) as exc:
            return _failed_quality(str(exc))

        # A run currently produces one visualization, so compare it with the
        # first reference visualization declared by the benchmark case.
        reference_path = Path(expected_visualizations[0].image_path)
        if not reference_path.is_absolute():
            reference_path = self.artifact_base_path / reference_path
        reference_image = _load_png(
            reference_path,
            description="Ground-truth visualization image",
        )

        judge = Agent(
            self.model,
            output_type=VisualizationEvaluation,
            instructions=_RUBRIC,
            model_settings=ModelSettings(temperature=0.0),
        )
        result = await judge.run(
            [
                "Generated plot (first image):",
                generated_image,
                "Ground-truth plot (second image):",
                reference_image,
            ]
        )
        assessment = result.output

        rationale = assessment.rationale
        return {
            "data_correct": EvaluationReason(
                value=assessment.data_correct,
                reason=rationale,
            ),
            "encoding_correct": EvaluationReason(
                value=assessment.encoding_correct,
                reason=rationale,
            ),
            "type_correct": EvaluationReason(
                value=assessment.type_correct,
                reason=rationale,
            ),
            "aesthetics_correct": EvaluationReason(
                value=assessment.aesthetics_correct,
                reason=rationale,
            ),
        }
