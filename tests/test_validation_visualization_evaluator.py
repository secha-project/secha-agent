from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel
from pydantic_evals.evaluators import EvaluationReason

from src.validation.eval_models import VisualizationOutput
from src.validation.evaluators.visualization import (
    ValidateVisualizationIncluded,
    ValidateVisualizationQuality,
    VisualizationEvaluation,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"test image payload"


def test_visualization_evaluation_rejects_non_boolean_scores() -> None:
    with pytest.raises(ValidationError):
        VisualizationEvaluation(
            data_correct=cast(Any, 1),
            encoding_correct=True,
            type_correct=True,
            aesthetics_correct=True,
            rationale="Invalid data correctness score.",
        )


def test_visualization_quality_uses_structured_multimodal_judgment(tmp_path) -> None:
    generated_path = tmp_path / "generated.png"
    reference_path = tmp_path / "reference.png"
    generated_path.write_bytes(PNG_BYTES)
    reference_path.write_bytes(PNG_BYTES)

    evaluator = ValidateVisualizationQuality(
        artifact_base_path=tmp_path,
        model=TestModel(
            custom_output_args={
                "data_correct": False,
                "encoding_correct": False,
                "type_correct": True,
                "aesthetics_correct": True,
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

    output = cast(
        dict[str, EvaluationReason],
        evaluator.evaluate_sync(cast(Any, ctx)),
    )

    assert output["data_correct"].value is False
    assert output["encoding_correct"].value is False
    assert output["type_correct"].value is True
    assert output["aesthetics_correct"].value is True
    assert output["encoding_correct"].reason == (
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

    output = cast(
        dict[str, EvaluationReason],
        evaluator.evaluate_sync(cast(Any, ctx)),
    )

    assert set(output) == {
        "data_correct",
        "encoding_correct",
        "type_correct",
        "aesthetics_correct",
    }
    assert all(result.value is False for result in output.values())


def test_visualization_included_requires_a_nonempty_image_file(tmp_path) -> None:
    invalid_path = tmp_path / "generated.png"
    invalid_path.write_bytes(b"")
    expected_output = [VisualizationOutput(image_path="reference.png")]
    evaluator = ValidateVisualizationIncluded()

    invalid_ctx = SimpleNamespace(
        expected_output=expected_output,
        attributes={"visualization_image_path": str(invalid_path)},
    )
    assert evaluator.evaluate_sync(cast(Any, invalid_ctx)) == {
        "graph_included": False
    }

    invalid_path.write_bytes(PNG_BYTES)
    assert evaluator.evaluate_sync(cast(Any, invalid_ctx)) == {
        "graph_included": True
    }


def test_visualization_quality_rejects_invalid_ground_truth_image(tmp_path) -> None:
    generated_path = tmp_path / "generated.png"
    reference_path = tmp_path / "reference.png"
    generated_path.write_bytes(PNG_BYTES)
    reference_path.write_bytes(b"")
    evaluator = ValidateVisualizationQuality(
        artifact_base_path=tmp_path,
        model=TestModel(),
    )
    ctx = SimpleNamespace(
        expected_output=[VisualizationOutput(image_path=reference_path.name)],
        attributes={"visualization_image_path": str(generated_path)},
    )

    with pytest.raises(ValueError, match="Ground-truth visualization image"):
        evaluator.evaluate_sync(cast(Any, ctx))
