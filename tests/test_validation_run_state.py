import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

import src.validation.run_benchmark as cli
from src.agent.config import AgentConfig
from src.agent.deps import Deps
from src.validation import runner
from src.validation.config import BenchmarkRunConfig
from src.validation.eval_models import BenchmarkCase, BenchmarkDataset
from src.validation.run_state import track_run


@pytest.fixture
def saved_run(monkeypatch, tmp_path):
    config = BenchmarkRunConfig(dataset="test", repetitions=3, max_concurrency=1)
    dataset = BenchmarkDataset(
        name="test", cases=[BenchmarkCase(name="task_1", inputs="question")]
    )
    agent = SimpleNamespace(run=None)
    deps = Deps(
        spark=cast(Any, object()),
        catalog=cast(Any, object()),
        schema_name="test",
        table_column_page_size=20,
    )
    monkeypatch.setattr(cli, "load_run_config", lambda _: ("test", config))
    monkeypatch.setattr(runner, "load_dataset", lambda _: dataset)
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(runner, "configure_observability", lambda _: None)
    monkeypatch.setattr(runner, "create_agent", lambda _: agent)
    monkeypatch.setattr(runner, "create_dependencies", lambda _: deps)
    monkeypatch.setattr(AgentConfig, "create_model", lambda _: object())
    reports = []

    def save(report, path):
        reports.append(report)
        path.write_text("saved report")
        return path

    monkeypatch.setattr(runner, "save_report", save)
    path = tmp_path / "queue/test.json"
    return SimpleNamespace(
        path=path,
        args=["--config", "test", "--run-state", str(path)],
        agent=agent,
        deps=deps,
        reports=reports,
    )


def answer(text):
    return SimpleNamespace(output=text, usage=SimpleNamespace(tool_calls=0))


def test_restart_keeps_completed_attempts_and_then_skips_finished_run(
    saved_run, monkeypatch
):
    calls = []

    class SimulatedInterruption(BaseException):
        pass

    async def interrupted(prompt, **_kwargs):
        calls.append(prompt)
        if len(calls) == 2:
            raise SimulatedInterruption()
        if len(calls) > 2:
            await asyncio.Event().wait()
        return answer("before interruption")

    saved_run.agent.run = interrupted
    with pytest.raises(BaseExceptionGroup) as error:
        cli.main(saved_run.args)
    assert error.value.subgroup(SimulatedInterruption) is not None
    state = json.loads(saved_run.path.read_text())
    directory = Path(state["run_directory"])
    assert not state["completed"]
    checkpoint = json.loads((directory / "checkpoint.json").read_text())
    assert [case["name"] for case in checkpoint["cases"]] == ["task_1 [1/3]"]
    retained = directory / "artifacts/retained.txt"
    retained.write_text("keep this artifact")
    resumed_calls = []

    async def resumed(prompt, **_kwargs):
        resumed_calls.append(prompt)
        return answer("after interruption")

    saved_run.agent.run = resumed
    cli.main(saved_run.args)
    assert resumed_calls == ["question", "question"]
    assert [case.output for case in saved_run.reports[0].cases] == [
        "before interruption",
        "after interruption",
        "after interruption",
    ]
    assert retained.read_text() == "keep this artifact"
    assert json.loads(saved_run.path.read_text()) == state | {"completed": True}
    assert list(directory.parent.iterdir()) == [directory]

    forbidden = Mock(side_effect=AssertionError("Completed run restarted"))
    monkeypatch.setattr(cli, "run_benchmark", forbidden)
    cli.main(saved_run.args)
    forbidden.assert_not_called()


def test_setup_failure_can_resume_before_any_attempt_was_completed(
    saved_run, monkeypatch
):
    monkeypatch.setattr(
        runner,
        "create_dependencies",
        Mock(side_effect=RuntimeError("Spark unavailable")),
    )
    with pytest.raises(RuntimeError, match="Spark unavailable"):
        cli.main(saved_run.args)
    state = json.loads(saved_run.path.read_text())
    directory = Path(state["run_directory"])
    checkpoint = json.loads((directory / "checkpoint.json").read_text())
    assert checkpoint["cases"] == checkpoint["failures"] == []

    async def run(_prompt, **_kwargs):
        return answer("recovered")

    saved_run.agent.run = run
    monkeypatch.setattr(runner, "create_dependencies", lambda _: saved_run.deps)
    cli.main(saved_run.args)
    assert len(saved_run.reports[0].cases) == 3
    assert json.loads(saved_run.path.read_text()) == state | {"completed": True}
    assert list(directory.parent.iterdir()) == [directory]


@pytest.mark.parametrize(
    "content",
    [
        "invalid JSON",
        '{"config": "different"}',
        '{"config": "test", "completed": true}',
    ],
)
def test_invalid_state_never_starts_paid_work(saved_run, monkeypatch, content):
    saved_run.path.parent.mkdir()
    saved_run.path.write_text(content)
    forbidden = Mock(side_effect=AssertionError("Invalid state started a run"))
    monkeypatch.setattr(cli, "run_benchmark", forbidden)
    with pytest.raises(ValueError):
        cli.main(saved_run.args)
    forbidden.assert_not_called()
    assert saved_run.path.read_text() == content


def test_run_lock_blocks_duplicate_execution_and_releases_after_interruption(tmp_path):
    path = tmp_path / "state.json"
    with pytest.raises(KeyboardInterrupt), track_run(path, "test"):
        with (
            pytest.raises(ValueError, match="Another process"),
            track_run(path, "test"),
        ):
            pytest.fail("Concurrent run acquired the lock")
        raise KeyboardInterrupt()
    with track_run(path, "test") as state:
        assert state.config == "test"


@pytest.mark.parametrize(
    "other",
    [["--dry-run"], ["--resume", "run"], ["--redo", "task_1"], ["--redo-tables"]],
)
def test_run_state_cannot_be_combined_with_other_modes(other):
    with pytest.raises(SystemExit) as error:
        cli.parse_args(["--config", "test", "--run-state", "state.json", *other])
    assert error.value.code == 2
