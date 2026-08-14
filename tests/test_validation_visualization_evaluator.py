from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

from src.validation.eval_models import VisualizationOutput
from src.validation.evaluators.visualization import (
    ValidateVisualizationQuality,
    VisualizationEvaluation,
)


def test_visualization_evaluation_rejects_scores_outside_rubric() -> None:
    with pytest.raises(ValidationError):
        VisualizationEvaluation(
            visualization_type=3,
            data_encoding=2,
            data_transformation=2,
            aesthetics=2,
            is_blank=0,
            rationale="Invalid visualization type score.",
        )


def test_visualization_quality_uses_structured_multimodal_judgment(tmp_path) -> None:
    generated_path = tmp_path / "generated.png"
    reference_path = tmp_path / "reference.png"
    generated_path.write_bytes(b"generated image")
    reference_path.write_bytes(b"reference image")

    evaluator = ValidateVisualizationQuality(
        artifact_base_path=tmp_path,
        model=TestModel(
            custom_output_args={
                "visualization_type": 2,
                "data_encoding": 1,
                "data_transformation": 0,
                "aesthetics": 2,
                "is_blank": 1,
                "rationale": "The encoding differs and the generated plot is blank.",
            }
        ),
    )
    ctx = SimpleNamespace(
        expected_output=[
            VisualizationOutput(
                spec_path="reference.json",
                image_path=reference_path.name,
            )
        ],
        attributes={"visualization_image_path": str(generated_path)},
    )

    output = evaluator.evaluate_sync(ctx)

    assert output["visualization_type"].value == 1.0
    assert output["data_encoding"].value == 0.5
    assert output["data_transformation"].value == 0.0
    assert output["aesthetics"].value == 1.0
    assert output["visualization_not_blank"].value is False
    assert output["data_encoding"].reason == (
        "The encoding differs and the generated plot is blank."
    )


def test_visualization_quality_skips_judge_when_generated_image_is_missing(
    tmp_path,
) -> None:
    evaluator = ValidateVisualizationQuality(
        artifact_base_path=tmp_path,
        model=TestModel(),
    )
    ctx = SimpleNamespace(
        expected_output=[
            VisualizationOutput(
                spec_path="reference.json",
                image_path="reference.png",
            )
        ],
        attributes={},
    )

    assert evaluator.evaluate_sync(ctx) == {}
