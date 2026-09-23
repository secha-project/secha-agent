from types import SimpleNamespace
from typing import Any, cast

import src.validation.runner as runner_module
from src.agent.deps import Deps
from src.agent.outputs.artifacts import Artifact, ArtifactCollector, TableArtifact
from src.validation.runner import AgentTask


def test_agent_task_uses_fresh_output_state_for_each_run(
    monkeypatch: Any,
) -> None:
    template_artifacts = ArtifactCollector()
    deps = Deps(
        spark=cast(Any, object()),
        catalog=cast(Any, object()),
        schema_name="bronze",
        table_column_page_size=20,
        artifacts=template_artifacts,
    )
    contexts: list[ArtifactCollector] = []
    table_output_stores = []
    recorded_metrics: list[tuple[str, int]] = []

    def run_sync(_prompt: str, **kwargs: Any) -> SimpleNamespace:
        context = kwargs["deps"].artifacts
        contexts.append(context)
        table_output_stores.append(kwargs["deps"].table_outputs)
        assert kwargs["deps"].table_artifact_max_rows is None
        return SimpleNamespace(output="answer", usage=SimpleNamespace(tool_calls=2))

    monkeypatch.setattr(
        runner_module,
        "increment_eval_metric",
        lambda name, value: recorded_metrics.append((name, value)),
    )

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
    assert template_artifacts not in contexts
    assert table_output_stores[0] is not table_output_stores[1]
    assert recorded_metrics == [("tool_calls", 2), ("tool_calls", 2)]


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
    written: list[tuple[Artifact, ...]] = []

    def run_sync(_prompt: str, **kwargs: Any) -> SimpleNamespace:
        artifact = make_table_artifact([{"value": 1}])
        kwargs["deps"].artifacts.publish(artifact)
        return SimpleNamespace(output="answer", usage=SimpleNamespace(tool_calls=0))

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
    assert isinstance(written[0][0], TableArtifact)
