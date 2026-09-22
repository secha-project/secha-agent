import asyncio
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from pydantic_evals.reporting import ReportCase

from src.validation.checkpoint import EvaluationCheckpoint
from src.validation.config import BenchmarkRunConfig
from src.validation.eval_models import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkOutput,
    CaseMetadata,
    TaskCategory,
)
from src.validation.runner import _copy_run_for_redo, _expand_attempts, evaluate


def _case(name: str) -> BenchmarkCase:
    return BenchmarkCase(
        name=name,
        inputs="question",
        metadata=CaseMetadata(
            case_id=uuid4(),
            query_code="query",
            category=TaskCategory.DATA_DISCOVERY,
        ),
        expected_output=None,
    )


def test_expand_attempts_skips_individual_completed_repetitions() -> None:
    dataset = BenchmarkDataset(name="benchmark", cases=[_case("task_1"), _case("task_2")])

    pending, attempts, names = _expand_attempts(
        dataset,
        repetitions=3,
        completed_names={"task_1 [1/3]", "task_2 [2/3]"},
    )

    assert names == [
        "task_1 [1/3]",
        "task_1 [2/3]",
        "task_1 [3/3]",
        "task_2 [1/3]",
        "task_2 [2/3]",
        "task_2 [3/3]",
    ]
    assert [case.name for case in pending.cases] == [
        "task_1 [2/3]",
        "task_1 [3/3]",
        "task_2 [1/3]",
        "task_2 [3/3]",
    ]
    assert attempts["task_1 [2/3]"] == ("task_1", 2)
    assert attempts["task_2 [3/3]"] == ("task_2", 3)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    metadata = {"run_config": "test"}
    checkpoint = EvaluationCheckpoint(
        checkpoint_path,
        name="test",
        experiment_metadata=metadata,
    )
    case_metadata = CaseMetadata(
        case_id=uuid4(),
        query_code="query",
        category=TaskCategory.DATA_DISCOVERY,
    )
    result = ReportCase[str, BenchmarkOutput, CaseMetadata](
        name="task_1 [1/2]",
        inputs="question",
        metadata=case_metadata,
        expected_output=None,
        output="answer",
        metrics={},
        attributes={},
        scores={},
        labels={},
        assertions={},
        task_duration=1.0,
        total_duration=1.0,
        source_case_name="task_1",
    )

    asyncio.run(checkpoint.record(result))
    resumed = EvaluationCheckpoint(
        checkpoint_path,
        name="test",
        experiment_metadata=metadata,
    )

    assert resumed.completed_names == {"task_1 [1/2]"}
    report = resumed.completed_report(
        ["task_1 [1/2]"],
        trace_id=None,
        span_id=None,
    )
    assert report.cases[0].source_case_name == "task_1"
    assert report.cases[0].metadata == case_metadata


def test_checkpoint_invalidate_removes_only_selected_attempts(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    metadata = {"run_config": "test"}
    checkpoint = EvaluationCheckpoint(
        checkpoint_path,
        name="test",
        experiment_metadata=metadata,
    )
    case_metadata = CaseMetadata(
        case_id=uuid4(),
        query_code="query",
        category=TaskCategory.DATA_DISCOVERY,
    )
    for name in ("task_1 [1/2]", "task_1 [2/2]", "task_2 [1/2]"):
        asyncio.run(
            checkpoint.record(
                ReportCase[str, BenchmarkOutput, CaseMetadata](
                    name=name,
                    inputs="question",
                    metadata=case_metadata,
                    expected_output=None,
                    output="answer",
                    metrics={},
                    attributes={},
                    scores={},
                    labels={},
                    assertions={},
                    task_duration=1.0,
                    total_duration=1.0,
                )
            )
        )

    invalidated = checkpoint.invalidate({"task_1 [1/2]", "task_1 [2/2]"})

    assert {result.name for result in invalidated} == {
        "task_1 [1/2]",
        "task_1 [2/2]",
    }
    resumed = EvaluationCheckpoint(
        checkpoint_path,
        name="test",
        experiment_metadata=metadata,
    )
    assert resumed.completed_names == {"task_2 [1/2]"}


def test_checkpoint_relocates_artifacts_in_a_copied_run(tmp_path: Path) -> None:
    source_artifacts = tmp_path / "source" / "artifacts"
    destination_artifacts = tmp_path / "destination" / "artifacts"
    source_artifacts.mkdir(parents=True)
    destination_artifacts.mkdir(parents=True)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = EvaluationCheckpoint(
        checkpoint_path,
        name="test",
        experiment_metadata={"run_config": "test"},
    )
    result = ReportCase[str, BenchmarkOutput, CaseMetadata](
        name="task_1",
        inputs="question",
        metadata=CaseMetadata(
            case_id=uuid4(),
            query_code="query",
            category=TaskCategory.DATA_DISCOVERY,
        ),
        expected_output=None,
        output="answer",
        metrics={},
        attributes={
            "artifact_paths": [str(source_artifacts / "task_1.csv")],
            "csv_path": str(source_artifacts / "task_1.csv"),
        },
        scores={},
        labels={},
        assertions={},
        task_duration=1.0,
        total_duration=1.0,
    )
    asyncio.run(checkpoint.record(result))

    checkpoint.relocate_artifacts(source_artifacts, destination_artifacts)

    assert checkpoint.report.cases[0].attributes == {
        "artifact_paths": [str(destination_artifacts / "task_1.csv")],
        "csv_path": str(destination_artifacts / "task_1.csv"),
    }


def test_copy_run_for_redo_starts_without_an_old_report(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "artifacts").mkdir(parents=True)
    (destination / "artifacts").mkdir(parents=True)
    (source / "checkpoint.json").write_text("checkpoint")
    (source / "report.json").write_text("old report")
    (source / "artifacts" / "task_1.csv").write_text("value\n1\n")

    _copy_run_for_redo(source, destination)

    assert (destination / "checkpoint.json").read_text() == "checkpoint"
    assert (destination / "artifacts" / "task_1.csv").read_text() == "value\n1\n"
    assert not (destination / "report.json").exists()


def test_evaluate_resumes_individual_repetitions(tmp_path: Path) -> None:
    dataset = BenchmarkDataset(name="benchmark", cases=[_case("task_1")])
    run_config = BenchmarkRunConfig(
        dataset="benchmark",
        repetitions=3,
        max_concurrency=2,
    )
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    checkpoint_path = tmp_path / "checkpoint.json"
    calls: list[str] = []

    def task(prompt: str) -> str:
        calls.append(prompt)
        return "answer"

    report = evaluate(
        dataset,
        cast(Any, task),
        "test-run",
        run_config,
        artifacts_dir,
        checkpoint_path,
    )

    assert calls == ["question"] * 3
    assert [case.name for case in report.cases] == [
        "task_1 [1/3]",
        "task_1 [2/3]",
        "task_1 [3/3]",
    ]
    assert {case.source_case_name for case in report.cases} == {"task_1"}

    def should_not_run(_: str) -> str:
        raise AssertionError("completed repetitions must be skipped")

    resumed = evaluate(
        dataset,
        cast(Any, should_not_run),
        "test-run",
        run_config,
        artifacts_dir,
        checkpoint_path,
    )

    assert [case.name for case in resumed.cases] == [
        "task_1 [1/3]",
        "task_1 [2/3]",
        "task_1 [3/3]",
    ]


def test_evaluate_redoes_every_repetition_of_selected_tasks(tmp_path: Path) -> None:
    dataset = BenchmarkDataset(
        name="benchmark",
        cases=[_case("task_1"), _case("task_2")],
    )
    run_config = BenchmarkRunConfig(
        dataset="benchmark",
        repetitions=2,
        max_concurrency=1,
    )
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    checkpoint_path = tmp_path / "checkpoint.json"
    first_calls: list[str] = []

    evaluate(
        dataset,
        cast(Any, lambda prompt: first_calls.append(prompt) or "old answer"),
        "test-run",
        run_config,
        artifacts_dir,
        checkpoint_path,
    )
    assert first_calls == ["question"] * 4

    redo_calls: list[str] = []
    report = evaluate(
        dataset,
        cast(Any, lambda prompt: redo_calls.append(prompt) or "new answer"),
        "test-run",
        run_config,
        artifacts_dir,
        checkpoint_path,
        redo_tasks=["task_1"],
    )

    assert redo_calls == ["question"] * 2
    outputs = {case.name: case.output for case in report.cases}
    assert outputs == {
        "task_1 [1/2]": "new answer",
        "task_1 [2/2]": "new answer",
        "task_2 [1/2]": "old answer",
        "task_2 [2/2]": "old answer",
    }


def test_evaluate_rejects_unknown_redo_task_without_changing_checkpoint(
    tmp_path: Path,
) -> None:
    dataset = BenchmarkDataset(name="benchmark", cases=[_case("task_1")])
    run_config = BenchmarkRunConfig(dataset="benchmark")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    checkpoint_path = tmp_path / "checkpoint.json"

    evaluate(
        dataset,
        cast(Any, lambda _prompt: "answer"),
        "test-run",
        run_config,
        artifacts_dir,
        checkpoint_path,
    )

    try:
        evaluate(
            dataset,
            cast(Any, lambda _prompt: "replacement"),
            "test-run",
            run_config,
            artifacts_dir,
            checkpoint_path,
            redo_tasks=["missing"],
        )
    except ValueError as error:
        assert "unknown benchmark task(s): missing" in str(error)
    else:
        raise AssertionError("Expected an unknown redo task to be rejected")

    resumed = EvaluationCheckpoint(
        checkpoint_path,
        name="test-run",
        experiment_metadata={
            "run_config": "test-run",
            "agent": run_config.agent.experiment_metadata(),
            "benchmark": run_config.model_dump(mode="json", exclude={"agent"}),
        },
    )
    assert resumed.completed_names == {"task_1"}
