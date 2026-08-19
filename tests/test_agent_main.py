import asyncio
import re
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

import src.agent.main as main_module
from src.agent.artifacts.base import (
    Artifact,
    ArtifactRef,
    ArtifactRegistry,
    RunArtifacts,
    UnknownArtifactError,
)
from src.agent.artifacts.table import TableContent
from src.agent.config import (
    DEFAULT_ENABLED_CAPABILITIES,
    AgentConfig,
    OpenAIProviderConfig,
    TuniProviderConfig,
)
from src.agent.main import AgentOutcome


def test_artifact_registry_keeps_logical_artifacts_addressable() -> None:
    registry = ArtifactRegistry()
    first = SimpleNamespace(kind="table")
    second = SimpleNamespace(kind="table")

    first_ref = registry.create(cast(Any, first))
    second_ref = registry.create(cast(Any, second))

    assert re.fullmatch(r"[0-9a-f]{32}", first_ref.artifact_id)
    assert re.fullmatch(r"[0-9a-f]{32}", second_ref.artifact_id)
    assert first_ref != second_ref
    assert registry.resolve(first_ref).source is first
    assert registry.resolve(second_ref).source is second

    with pytest.raises(UnknownArtifactError):
        registry.resolve(ArtifactRef(artifact_id="stale"))


def test_registered_artifacts_are_immutable_and_content_is_registry_controlled(
    make_table_artifact: Any,
) -> None:
    registry = ArtifactRegistry()
    ref = ArtifactRef(artifact_id="result")
    artifact = Artifact(ref=ref, source=cast(Any, SimpleNamespace(kind="table")))
    content = make_table_artifact([{"value": 1}]).require_content()
    registry.add(artifact)

    updated = registry.set_content(ref, content)

    assert artifact.content is None
    assert registry.resolve(ref) is updated
    with pytest.raises(FrozenInstanceError):
        artifact.content = None  # ty: ignore[invalid-assignment]


def test_agent_config_keeps_immutable_valid_settings() -> None:
    config = AgentConfig(
        _env_file=None,
        catalog_url="https://catalog.example",
        catalog_token="token",
        spark_url="sc://spark.example",
        model_provider="openai",
        enabled_capabilities=("catalog", "catalog", "query"),
        otel_exporter_otlp_endpoint="http://otel.example",
    )

    assert config.model_provider == "openai"
    assert config.enabled_capabilities == frozenset({"catalog", "query"})
    with pytest.raises(ValidationError, match="Instance is frozen"):
        config.request_limit = 0


def test_agent_config_uses_default_capabilities() -> None:
    config = AgentConfig(
        _env_file=None,
        catalog_url="https://catalog.example",
        catalog_token="token",
        spark_url="sc://spark.example",
        otel_exporter_otlp_endpoint="http://otel.example",
    )

    assert config.enabled_capabilities == DEFAULT_ENABLED_CAPABILITIES


def test_agent_config_rejects_invalid_cross_field_state() -> None:
    with pytest.raises(
        ValidationError,
        match="semantic layer capability requires query",
    ):
        AgentConfig(
            _env_file=None,
            catalog_url="https://catalog.example",
            catalog_token="token",
            spark_url="sc://spark.example",
            model_provider="openai",
            enabled_capabilities=("semantic_layer",),
            otel_exporter_otlp_endpoint="http://otel.example",
        )


def test_agent_config_loads_only_the_selected_provider_from_environment(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("TUNI_INFERENCE_URL", "https://tuni.example")
    monkeypatch.setenv("TUNI_INFERENCE_API_KEY", "secret")
    monkeypatch.setenv("TUNI_INFERENCE_MODEL_NAME", "custom-model")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-openai-secret")

    config = AgentConfig(
        _env_file=None,
        catalog_url="https://catalog.example",
        catalog_token="token",
        spark_url="sc://spark.example",
        model_provider="tuni",
        enabled_capabilities=("query",),
        otel_exporter_otlp_endpoint="http://otel.example",
    )
    provider = config.load_provider()

    assert isinstance(provider, TuniProviderConfig)
    assert provider.url == "https://tuni.example"
    assert provider.model_name == "custom-model"
    assert provider.api_key.get_secret_value() == "secret"
    assert "secret" not in repr(provider)
    assert "unused-openai-secret" not in repr(config)

    openai_config = AgentConfig(
        _env_file=None,
        catalog_url="https://catalog.example",
        catalog_token="token",
        spark_url="sc://spark.example",
        model_provider="openai",
        enabled_capabilities=("query",),
        otel_exporter_otlp_endpoint="http://otel.example",
    )
    openai_provider = openai_config.load_provider()

    assert isinstance(openai_provider, OpenAIProviderConfig)
    assert openai_provider.api_key.get_secret_value() == "unused-openai-secret"
    assert "https://tuni.example" not in repr(openai_config)


def test_agent_config_accepts_a_per_run_model_override() -> None:
    config = AgentConfig(
        _env_file=None,
        catalog_url="https://catalog.example",
        catalog_token="token",
        spark_url="sc://spark.example",
        model_provider="tuni",
        model_name="experiment-model",
        enabled_capabilities=("query",),
        otel_exporter_otlp_endpoint="http://otel.example",
    )

    assert config.model_provider == "tuni"
    assert config.model_name == "experiment-model"


def test_artifact_source_can_materialize_from_an_artifact_reference(
    make_table_artifact: Any,
) -> None:
    artifacts = RunArtifacts()
    input_artifact = make_table_artifact([{"value": 1}])
    artifacts.registry.add(input_artifact)

    class ReferencedTableSource:
        kind = "table"

        def __init__(self, input_ref: ArtifactRef) -> None:
            self.input_ref = input_ref

        async def materialize(self, artifacts: RunArtifacts) -> TableContent:
            content = await artifacts.materialize(self.input_ref)
            assert isinstance(content, TableContent)
            return content

    output_ref = artifacts.create(cast(Any, ReferencedTableSource(input_artifact.ref)))

    assert asyncio.run(artifacts.materialize(output_ref)) is input_artifact.content
    assert artifacts.registry.resolve(output_ref).ref == output_ref


def test_run_uses_run_scoped_artifacts(
    monkeypatch: Any,
) -> None:
    config = cast(
        AgentConfig,
        SimpleNamespace(
            catalog_url="https://catalog.example",
            catalog_token="token",
            catalog_name="analytics",
            schema_name="bronze",
            table_column_page_size=20,
            temperature=0.0,
            request_limit=3,
            enabled_capabilities=("catalog", "query"),
            create_model=lambda: "model",
        ),
    )
    spark = object()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(main_module, "get_spark_session", lambda _config: spark)
    def run_sync(prompt: str, **kwargs: Any) -> SimpleNamespace:
        captured["prompt"] = prompt
        captured.update(kwargs)
        return SimpleNamespace(output="answer")

    monkeypatch.setattr(
        main_module,
        "create_agent",
        lambda capabilities: SimpleNamespace(run_sync=run_sync),
    )

    outcome = main_module.run("question", config)

    assert outcome == AgentOutcome(output="answer", artifacts=())
    assert captured["prompt"] == "question"
    assert captured["deps"].spark is spark
    assert isinstance(captured["deps"].artifacts, RunArtifacts)


def test_main_accepts_a_prompt_argument(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    monkeypatch.setattr(
        main_module,
        "run",
        lambda prompt: AgentOutcome(output=f"answer to {prompt}", artifacts=()),
    )

    main_module.main(["question"])

    assert capsys.readouterr().out == "answer to question\n"


def test_run_returns_artifacts_published_during_the_one_off_run(
    monkeypatch: Any,
    make_table_artifact: Any,
) -> None:
    config = cast(
        AgentConfig,
        SimpleNamespace(
            catalog_url="https://catalog.example",
            catalog_token="token",
            catalog_name="analytics",
            schema_name="bronze",
            table_column_page_size=20,
            temperature=0.0,
            request_limit=3,
            enabled_capabilities=("query",),
            create_model=lambda: "model",
        ),
    )
    monkeypatch.setattr(main_module, "get_spark_session", lambda _config: object())
    def run_sync(_prompt: str, **kwargs: Any) -> SimpleNamespace:
        artifact = make_table_artifact([{"value": 1}])
        context = kwargs["deps"].artifacts
        context.registry.add(artifact)
        asyncio.run(context.publish(artifact.ref))
        return SimpleNamespace(output="answer")

    monkeypatch.setattr(
        main_module,
        "create_agent",
        lambda capabilities: SimpleNamespace(run_sync=run_sync),
    )

    outcome = main_module.run("question", config)

    assert outcome.output == "answer"
    assert len(outcome.artifacts) == 1
    assert outcome.artifacts[0].kind == "table"
