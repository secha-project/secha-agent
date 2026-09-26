from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from pydantic_ai import BinaryContent, capture_run_messages
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.test import TestModel
from pydantic_evals.evaluators import EvaluationReason

from src.agent.config import OpenAIProviderConfig
from src.validation.eval_models import VisualizationOutput
from src.validation.evaluators import visualization
from src.validation.evaluators.visualization import (
    ValidateVisualizationIncluded,
    ValidateVisualizationQuality,
    VisualizationEvaluation,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"test image payload"


@pytest.mark.parametrize("environment_key", [None, "environment-test-key"])
def test_default_judge_loads_credentials_and_uses_max_reasoning(
    tmp_path, monkeypatch, environment_key,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=dotenv-test-key\n")
    monkeypatch.setitem(OpenAIProviderConfig.model_config, "env_file", env_file)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if environment_key:
        monkeypatch.setenv("OPENAI_API_KEY", environment_key)

    image_path = tmp_path / "plot.png"
    image_path.write_bytes(PNG_BYTES)
    assessment = VisualizationEvaluation(
        data_correct=True,
        encoding_correct=True,
        type_correct=True,
        aesthetics_correct=True,
        rationale="The plots match.",
    )
    run = AsyncMock(return_value=SimpleNamespace(output=assessment))

    def create_judge(model, **kwargs):
        assert isinstance(model, OpenAIResponsesModel)
        assert model.model_name == "gpt-6-luna"
        assert model.client.api_key == (environment_key or "dotenv-test-key")
        assert model.client.max_retries == 0
        assert kwargs["model_settings"] == {"openai_reasoning_effort": "max"}
        return SimpleNamespace(run=run)

    monkeypatch.setattr(visualization, "Agent", create_judge)
    evaluator = ValidateVisualizationQuality(artifact_base_path=tmp_path)
    ctx = SimpleNamespace(
        inputs="Plot hourly mean power in kW for March 1, 2025 UTC.",
        expected_output=[VisualizationOutput(image_path=image_path.name)],
        attributes={"visualization_image_path": str(image_path)},
    )
    output = cast(dict[str, EvaluationReason], evaluator.evaluate_sync(cast(Any, ctx)))
    assert all(result.value is True for result in output.values())
    run.assert_awaited_once()


def test_visualization_evaluation_rejects_non_boolean_scores() -> None:
    with pytest.raises(ValidationError):
        VisualizationEvaluation(
            data_correct=cast(Any, 1),
            encoding_correct=True,
            type_correct=True,
            aesthetics_correct=True,
            rationale="Invalid data correctness score.",
        )


@pytest.mark.parametrize(
    "task_text",
    [
        "Plot hourly mean power in kW for March 1, 2025 UTC.",
        "Compare L1, L2, and L3 voltage in V.\nUse a line chart over 12:00–13:00 UTC.",
    ],
)
def test_visualization_quality_uses_task_and_structured_multimodal_judgment(
    tmp_path, task_text
) -> None:
    generated_path = tmp_path / "generated.png"
    reference_path = tmp_path / "reference.png"
    generated_path.write_bytes(PNG_BYTES + b"generated")
    reference_path.write_bytes(PNG_BYTES + b"reference")

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
        inputs=task_text,
        expected_output=[
            VisualizationOutput(
                spec_path="reference.json",
                image_path=reference_path.name,
            )
        ],
        attributes={"visualization_image_path": str(generated_path)},
    )

    with capture_run_messages() as messages:
        output = cast(
            dict[str, EvaluationReason],
            evaluator.evaluate_sync(cast(Any, ctx)),
        )

    request = next(message for message in messages if isinstance(message, ModelRequest))
    assert request.instructions == visualization._RUBRIC.strip()
    prompt = next(
        part.content for part in request.parts if isinstance(part, UserPromptPart)
    )
    assert isinstance(prompt, list)
    assert len(prompt) == 5
    assert prompt[0] == f"Original task:\n{task_text}"
    assert prompt[1] == "Generated plot (first image):"
    assert isinstance(prompt[2], BinaryContent)
    assert prompt[2].data == generated_path.read_bytes()
    assert prompt[2].media_type == "image/png"
    assert prompt[2].vendor_metadata == {"detail": "high"}
    assert prompt[3] == "Ground-truth plot (second image):"
    assert isinstance(prompt[4], BinaryContent)
    assert prompt[4].data == reference_path.read_bytes()
    assert prompt[4].media_type == "image/png"
    assert prompt[4].vendor_metadata == {"detail": "high"}

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
