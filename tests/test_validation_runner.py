from types import SimpleNamespace
from typing import Any, cast

import pytest

import src.validation.runner as runner_module
from src.agent.deps import Deps
from src.agent.query_results import (
    QueryResultRegistry,
    UnknownQueryResultReferenceError,
)
from src.validation.runner import AgentTask


def test_agent_task_uses_a_fresh_result_registry_for_each_run(
    monkeypatch: Any,
) -> None:
    template_registry = QueryResultRegistry()
    deps = Deps(
        spark=cast(Any, object()),
        catalog=cast(Any, object()),
        schema_name="bronze",
        table_column_page_size=20,
        result_registry=template_registry,
    )
    registries: list[QueryResultRegistry] = []
    references: list[str] = []

    def run_sync(_prompt: str, **kwargs: Any) -> SimpleNamespace:
        registry = kwargs["deps"].result_registry
        registries.append(registry)
        references.append(
            registry.create(cast(Any, object()))
        )
        return SimpleNamespace(output="answer")

    monkeypatch.setattr(runner_module.agent, "run_sync", run_sync)
    task = AgentTask(
        deps=deps,
        model=cast(Any, object()),
        model_settings=cast(Any, object()),
        usage_limits=cast(Any, object()),
    )

    assert task("first") == "answer"
    assert task("second") == "answer"

    assert registries[0] is not registries[1]
    assert template_registry not in registries
    assert references[0] != references[1]
    with pytest.raises(UnknownQueryResultReferenceError):
        registries[0].resolve(references[1])
