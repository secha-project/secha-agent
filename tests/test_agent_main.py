import asyncio
import re
from types import SimpleNamespace
from typing import Any, cast

import pytest

import src.agent.main as main_module
from src.agent.artifacts.base import (
    ArtifactContext,
    ArtifactRef,
    ArtifactRegistry,
    UnknownArtifactError,
)
from src.agent.artifacts.table import TableContent
from src.agent.config import AgentConfig
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
        registry.resolve(ArtifactRef(artifact_id="stale", artifact_kind="table"))


def test_artifact_source_can_materialize_from_an_artifact_reference(
    make_table_artifact: Any,
) -> None:
    context = ArtifactContext()
    input_artifact = make_table_artifact([{"value": 1}])
    context.registry.add(input_artifact)

    class ReferencedTableSource:
        kind = "table"

        def __init__(self, input_ref: ArtifactRef) -> None:
            self.input_ref = input_ref

        async def materialize(self, artifact_context: ArtifactContext) -> TableContent:
            return await artifact_context.materialize(self.input_ref)

    output_ref = context.create(cast(Any, ReferencedTableSource(input_artifact.ref)))

    assert asyncio.run(context.materialize(output_ref)) is input_artifact.content
    assert context.registry.resolve(output_ref).ref == output_ref


def test_run_uses_a_run_scoped_artifact_context(
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
        ),
    )
    spark = object()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(main_module, "get_spark_session", lambda _config: spark)
    monkeypatch.setattr(main_module, "create_model", lambda _config: "model")

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
    assert isinstance(captured["deps"].artifact_context, ArtifactContext)


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
        ),
    )
    monkeypatch.setattr(main_module, "get_spark_session", lambda _config: object())
    monkeypatch.setattr(main_module, "create_model", lambda _config: "model")

    def run_sync(_prompt: str, **kwargs: Any) -> SimpleNamespace:
        artifact = make_table_artifact([{"value": 1}])
        context = kwargs["deps"].artifact_context
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
