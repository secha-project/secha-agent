import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

import src.validation.runner as runner_module
from src.agent.artifacts.base import (
    ArtifactContext,
    UnknownArtifactError,
)
from src.agent.deps import Deps
from src.validation.runner import AgentTask


def test_agent_task_uses_a_fresh_artifact_context_for_each_run(
    monkeypatch: Any,
) -> None:
    template_context = ArtifactContext()
    deps = Deps(
        spark=cast(Any, object()),
        catalog=cast(Any, object()),
        schema_name="bronze",
        table_column_page_size=20,
        artifact_context=template_context,
    )
    contexts: list[ArtifactContext] = []
    references = []

    def run_sync(_prompt: str, **kwargs: Any) -> SimpleNamespace:
        context = kwargs["deps"].artifact_context
        contexts.append(context)
        references.append(context.create(cast(Any, SimpleNamespace(kind="table"))))
        return SimpleNamespace(output="answer")

    task = AgentTask(
        agent=cast(Any, SimpleNamespace(run_sync=run_sync)),
        deps=deps,
        model=cast(Any, object()),
        model_settings=cast(Any, object()),
        usage_limits=cast(Any, object()),
    )

    assert task("first") == "answer"
    assert task("second") == "answer"

    assert contexts[0] is not contexts[1]
    assert template_context not in contexts
    assert contexts[0].policy.table_max_rows is None
    assert references[0] != references[1]
    with pytest.raises(UnknownArtifactError):
        contexts[0].registry.resolve(references[1])


def test_agent_task_writes_only_the_artifacts_from_that_run(
    monkeypatch: Any,
    make_table_artifact: Any,
) -> None:
    deps = Deps(
        spark=cast(Any, object()),
        catalog=cast(Any, object()),
        schema_name="bronze",
        table_column_page_size=20,
    )
    written: list[tuple[object, ...]] = []

    def run_sync(_prompt: str, **kwargs: Any) -> SimpleNamespace:
        artifact = make_table_artifact([{"value": 1}])
        context = kwargs["deps"].artifact_context
        context.registry.add(artifact)
        asyncio.run(context.publish(artifact.ref))
        return SimpleNamespace(output="answer")

    monkeypatch.setattr(
        runner_module,
        "write_evaluation_artifacts",
        lambda artifacts: written.append(artifacts),
    )
    task = AgentTask(
        agent=cast(Any, SimpleNamespace(run_sync=run_sync)),
        deps=deps,
        model=cast(Any, object()),
        model_settings=cast(Any, object()),
        usage_limits=cast(Any, object()),
    )

    assert task("question") == "answer"
    assert len(written) == 1
    assert len(written[0]) == 1
    assert written[0][0].kind == "table"
