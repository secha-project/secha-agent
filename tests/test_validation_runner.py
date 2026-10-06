import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml

import src.validation.runner as runner_module
from src.agent.deps import Deps
from src.agent.outputs.artifacts import Artifact, ArtifactCollector, TableArtifact
from src.validation.runner import AgentTask


def test_load_dataset_preserves_paths_and_source() -> None:
    dataset_path = (
        runner_module.BENCHMARKS_DIR / "power-quality-monitoring" / "dataset.yaml"
    )
    original = dataset_path.read_bytes()
    source = yaml.safe_load(original)

    dataset = runner_module.load_dataset("power-quality-monitoring")

    assert [case.name for case in dataset.cases] == [
        case["name"] for case in source["cases"]
    ]
    for case, raw_case in zip(dataset.cases, source["cases"], strict=True):
        assert case.metadata is not None
        assert case.metadata.query_path == raw_case["metadata"].get("query_path")
        assert case.metadata.ggsql_path == raw_case["metadata"].get("ggsql_path")
        assert case.evaluators
    assert dataset.report_evaluators
    assert dataset_path.read_bytes() == original


def test_load_dataset_rejects_inline_sql(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "test"
    dataset_dir.mkdir()
    metadata = {
        "case_id": "11111111-1111-1111-1111-111111111111",
        "category": "data discovery",
        "query_code": "SELECT 'inline SQL'",
    }
    (dataset_dir / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "test",
                "cases": [
                    {
                        "name": "case",
                        "inputs": "question",
                        "metadata": metadata,
                        "expected_output": [{"text": ["answer"]}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner_module, "BENCHMARKS_DIR", tmp_path)

    with pytest.raises(ValueError, match="query_code"):
        runner_module.load_dataset("test")


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
    chart_output_stores = []
    recorded_metrics: list[tuple[str, int]] = []

    async def run(_prompt: str, **kwargs: Any) -> SimpleNamespace:
        context = kwargs["deps"].artifacts
        contexts.append(context)
        table_output_stores.append(kwargs["deps"].table_outputs)
        chart_output_stores.append(kwargs["deps"].chart_outputs)
        assert kwargs["deps"].table_artifact_max_rows is None
        return SimpleNamespace(output="answer", usage=SimpleNamespace(tool_calls=2))

    monkeypatch.setattr(
        runner_module,
        "increment_eval_metric",
        lambda name, value: recorded_metrics.append((name, value)),
    )

    task = AgentTask(
        agent=cast(Any, SimpleNamespace(run=run)),
        deps=deps,
        model=cast(Any, object()),
        model_settings=cast(Any, object()),
        usage_limits=cast(Any, object()),
    )

    assert asyncio.run(task("first")) == "answer"
    assert asyncio.run(task("second")) == "answer"

    assert contexts[0] is not contexts[1]
    assert template_artifacts not in contexts
    assert table_output_stores[0] is not table_output_stores[1]
    assert chart_output_stores[0] is not chart_output_stores[1]
    assert deps.chart_outputs not in chart_output_stores
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

    async def run(_prompt: str, **kwargs: Any) -> SimpleNamespace:
        artifact = make_table_artifact([{"value": 1}])
        kwargs["deps"].artifacts.publish(artifact)
        return SimpleNamespace(output="answer", usage=SimpleNamespace(tool_calls=0))

    monkeypatch.setattr(
        runner_module,
        "write_evaluation_artifacts",
        lambda artifacts: written.append(artifacts),
    )
    task = AgentTask(
        agent=cast(Any, SimpleNamespace(run=run)),
        deps=deps,
        model=cast(Any, object()),
        model_settings=cast(Any, object()),
        usage_limits=cast(Any, object()),
    )

    assert asyncio.run(task("question")) == "answer"
    assert len(written) == 1
    assert len(written[0]) == 1
    assert isinstance(written[0][0], TableArtifact)


def test_concurrent_benchmark_cases_share_one_model_event_loop(monkeypatch, tmp_path):
    from src.validation.config import BenchmarkRunConfig
    from src.validation.eval_models import BenchmarkCase, BenchmarkDataset

    monkeypatch.setattr(runner_module, "_experiment_metadata", lambda *_: {})
    loops = []
    active = 0
    peak_active = 0

    async def run(_prompt, **_kwargs):
        nonlocal active, peak_active
        loops.append(asyncio.get_running_loop())
        active += 1
        peak_active = max(active, peak_active)
        await asyncio.sleep(0.01)
        active -= 1
        return SimpleNamespace(output="answer", usage=SimpleNamespace(tool_calls=0))

    task = AgentTask(
        agent=cast(
            Any,
            SimpleNamespace(
                run=run,
                run_sync=lambda *args, **kwargs: asyncio.run(run(*args, **kwargs)),
            ),
        ),
        deps=Deps(
            spark=cast(Any, object()),
            catalog=cast(Any, object()),
            schema_name="bronze",
            table_column_page_size=20,
        ),
        model=cast(Any, object()),
        model_settings={},
        usage_limits=cast(Any, object()),
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    report = runner_module.evaluate(
        BenchmarkDataset(
            name="test",
            cases=[
                BenchmarkCase(name=f"case_{n}", inputs="question") for n in range(4)
            ],
        ),
        task,
        "test",
        BenchmarkRunConfig(dataset="test", max_concurrency=4),
        artifacts,
        tmp_path / "checkpoint.json",
    )
    assert len(report.cases) == 4
    assert report.failures == []
    assert len(set(loops)) == 1
    assert peak_active == 4


def test_resume_rejects_invalid_checkpoint_before_external_setup(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    from src.validation.checkpoint import EvaluationCheckpoint
    from src.validation.config import BenchmarkRunConfig
    from src.validation.eval_models import BenchmarkDataset

    config = BenchmarkRunConfig(dataset="benchmark")
    (tmp_path / "artifacts").mkdir()
    monkeypatch.setattr(
        runner_module,
        "load_dataset",
        lambda _: BenchmarkDataset(name="benchmark", cases=[]),
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Invalid resume must fail before external setup or copying")

    for name in (
        "configure_observability",
        "create_dependencies",
        "create_agent",
        "create_run_directory",
    ):
        monkeypatch.setattr(runner_module, name, forbidden)
    with pytest.raises(ValueError, match="no checkpoint"):
        runner_module.run_benchmark("test", config, resume_directory=tmp_path)

    checkpoint = EvaluationCheckpoint(
        tmp_path / "checkpoint.json", name="wrong-run", experiment_metadata={}
    )
    checkpoint.completed_report([], trace_id=None, span_id=None)
    with pytest.raises(ValueError, match="belongs to run"):
        runner_module.run_benchmark("test", config, resume_directory=tmp_path)

    checkpoint.report.name = "test"
    checkpoint.completed_report([], trace_id=None, span_id=None)
    with pytest.raises(ValueError, match="different run configuration"):
        runner_module.run_benchmark("test", config, resume_directory=tmp_path)

    (tmp_path / "checkpoint.json").write_text("invalid json")
    with pytest.raises(ValueError):
        runner_module.run_benchmark("test", config, resume_directory=tmp_path)
