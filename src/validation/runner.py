import asyncio
import os
import shutil
import traceback
from collections.abc import Collection
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import logfire
from pydantic_ai import Agent, ModelSettings, UsageLimits, capture_run_messages
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import Model
from pydantic_ai.retries import RetryConfig
from pydantic_evals import increment_eval_metric, set_eval_attribute
from pydantic_evals.evaluators import EvaluatorFailure, ReportEvaluatorContext
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

from src.agent.agent import create_agent
from src.agent.catalog import UnityCatalogClient
from src.agent.config import AgentConfig
from src.agent.deps import Deps
from src.agent.outputs.artifacts import ArtifactCollector
from src.agent.outputs.tables import TableOutputStore
from src.agent.spark_runtime import get_spark_session
from src.validation.checkpoint import EvaluationCheckpoint
from src.validation.config import BENCHMARKS_DIR, RESULTS_DIR, BenchmarkRunConfig
from src.validation.eval_models import (
    BenchmarkCase,
    BenchmarkDataset,
    NumberOutput,
    TableOutput,
    TextOutput,
    VisualizationOutput,
)
from src.validation.evaluators import (
    DEFAULT_REPORT_EVALUATORS,
    ValidateNumberOutput,
    ValidateTableIncluded,
    ValidateTableValues,
    ValidateTextOutput,
    ValidateVisualizationIncluded,
    ValidateVisualizationQuality,
)
from src.validation.lifecycle import LifecycleFactory, write_evaluation_artifacts
from src.validation.progress_log import ProgressLog
from src.validation.reporting import BenchmarkReport, save_report

CUSTOM_EVALUATOR_TYPES = (
    ValidateTextOutput,
    ValidateNumberOutput,
    ValidateTableIncluded,
    ValidateTableValues,
    ValidateVisualizationIncluded,
    ValidateVisualizationQuality,
)


@dataclass(frozen=True)
class AgentTask:
    agent: Agent[Deps, str]
    deps: Deps
    model: Model
    model_settings: ModelSettings
    usage_limits: UsageLimits
    capture_messages: bool = False

    async def __call__(self, prompt: str) -> str:
        artifacts = ArtifactCollector()
        run_deps = replace(
            self.deps,
            artifacts=artifacts,
            table_outputs=TableOutputStore(),
            table_artifact_max_rows=None,
        )
        with (
            capture_run_messages()
            if self.capture_messages
            else nullcontext() as messages
        ):
            try:
                result = await self.agent.run(
                    prompt,
                    deps=run_deps,
                    model=self.model,
                    model_settings=self.model_settings,
                    usage_limits=self.usage_limits,
                )
            finally:
                if messages is not None:
                    set_eval_attribute(
                        "agent_messages",
                        ModelMessagesTypeAdapter.dump_python(messages, mode="json"),
                    )
        increment_eval_metric("tool_calls", result.usage.tool_calls)
        write_evaluation_artifacts(artifacts.published)
        return result.output


def configure_observability(config: AgentConfig) -> None:
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = config.otel_exporter_otlp_endpoint
    logfire.configure(
        send_to_logfire=False,
        service_name=config.logfire_service_name,
        metrics=False,
    )
    logfire.instrument_pydantic_ai()


def create_dependencies(config: AgentConfig) -> Deps:
    spark = get_spark_session(config)
    return Deps(
        spark=spark,
        catalog=UnityCatalogClient(
            config.catalog_url,
            config.catalog_token,
            config.catalog_name,
        ),
        schema_name=config.schema_name,
        table_column_page_size=config.table_column_page_size,
        semantic_layer_enabled="semantic_layer" in config.enabled_capabilities,
    )


def load_dataset(dataset_name: str) -> BenchmarkDataset:
    dataset_path = BENCHMARKS_DIR / dataset_name / "dataset.yaml"
    dataset = BenchmarkDataset.from_file(
        dataset_path,
        custom_evaluator_types=CUSTOM_EVALUATOR_TYPES,
    )
    attach_case_evaluators(dataset, dataset_path.parent)
    dataset.report_evaluators.extend(
        evaluator_type() for evaluator_type in DEFAULT_REPORT_EVALUATORS
    )
    return dataset


def attach_case_evaluators(dataset: BenchmarkDataset, artifact_base_path: Path) -> None:
    for case in dataset.cases:
        expected_output = case.expected_output
        if not isinstance(expected_output, list):
            continue

        if any(isinstance(item, TextOutput) for item in expected_output):
            case.evaluators.append(ValidateTextOutput())

        if any(isinstance(item, NumberOutput) for item in expected_output):
            case.evaluators.append(ValidateNumberOutput())

        if any(isinstance(item, TableOutput) for item in expected_output):
            case.evaluators.extend(
                [
                    ValidateTableIncluded(),
                    ValidateTableValues(artifact_base_path=artifact_base_path),
                ]
            )

        if any(isinstance(item, VisualizationOutput) for item in expected_output):
            case.evaluators.extend(
                [
                    ValidateVisualizationIncluded(),
                    ValidateVisualizationQuality(artifact_base_path=artifact_base_path),
                ]
            )


def create_run_directory(run_name: str, dataset_name: str) -> Path:
    dataset_directory = RESULTS_DIR / dataset_name
    dataset_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).astimezone().strftime("%Y-%m-%d_%H-%M")
    directory_name = f"{run_name}_{timestamp}"
    suffix = 1
    while True:
        run_directory = dataset_directory / (
            directory_name if suffix == 1 else f"{directory_name}_{suffix}"
        )
        try:
            run_directory.mkdir()
        except FileExistsError:
            suffix += 1
        else:
            break
    (run_directory / "artifacts").mkdir()
    return run_directory


def _expand_attempts(
    dataset: BenchmarkDataset,
    repetitions: int,
    completed_names: set[str],
) -> tuple[BenchmarkDataset, dict[str, tuple[str, int]], list[str]]:
    """Turn repetitions into independently resumable, explicitly named cases."""
    pending_cases: list[BenchmarkCase] = []
    attempts: dict[str, tuple[str, int]] = {}
    attempt_names: list[str] = []

    for case in dataset.cases:
        if case.name is None:
            raise ValueError("Evaluation cases must have names to support resuming.")
        for repetition_index in range(1, repetitions + 1):
            attempt_name = (
                case.name
                if repetitions == 1
                else f"{case.name} [{repetition_index}/{repetitions}]"
            )
            attempt_names.append(attempt_name)
            if attempt_name in completed_names:
                continue
            pending_cases.append(
                BenchmarkCase(
                    name=attempt_name,
                    inputs=case.inputs,
                    metadata=case.metadata,
                    expected_output=case.expected_output,
                    evaluators=tuple(case.evaluators),
                )
            )
            if repetitions > 1:
                attempts[attempt_name] = (case.name, repetition_index)

    return (
        BenchmarkDataset(
            name=dataset.name,
            cases=pending_cases,
            evaluators=dataset.evaluators,
        ),
        attempts,
        attempt_names,
    )


def _task_attempt_names(task_name: str, repetitions: int) -> set[str]:
    if repetitions == 1:
        return {task_name}
    return {
        f"{task_name} [{repetition_index}/{repetitions}]"
        for repetition_index in range(1, repetitions + 1)
    }


def _validate_redo_tasks(
    dataset: BenchmarkDataset,
    redo_tasks: Collection[str],
) -> None:
    task_names = {case.name for case in dataset.cases if case.name is not None}
    unknown_redo_tasks = set(redo_tasks) - task_names
    if unknown_redo_tasks:
        unknown = ", ".join(sorted(unknown_redo_tasks))
        available = ", ".join(sorted(task_names))
        raise ValueError(
            f"Cannot redo unknown benchmark task(s): {unknown}. "
            f"Available tasks: {available}."
        )


def _copy_run_for_redo(source_directory: Path, destination_directory: Path) -> None:
    checkpoint_path = source_directory / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise ValueError(f"Redo source has no checkpoint: {checkpoint_path}")

    shutil.copy2(checkpoint_path, destination_directory / "checkpoint.json")
    shutil.copytree(
        source_directory / "artifacts",
        destination_directory / "artifacts",
        dirs_exist_ok=True,
    )


def _remove_invalidated_artifacts(
    results: Collection[object],
    artifacts_dir: Path,
) -> None:
    """Remove artifact files referenced only by attempts that will be redone."""
    resolved_artifacts_dir = artifacts_dir.resolve()
    for result in results:
        attributes = getattr(result, "attributes", {})
        if not isinstance(attributes, dict):
            continue

        artifact_paths = attributes.get("artifact_paths", [])
        if not isinstance(artifact_paths, list):
            continue
        for value in artifact_paths:
            if not isinstance(value, str):
                continue
            path = Path(value).resolve()
            if path.parent == resolved_artifacts_dir and path.is_file():
                path.unlink()


async def _apply_report_evaluators(
    report: BenchmarkReport,
    dataset: BenchmarkDataset,
) -> None:
    """Run whole-report evaluators after checkpointed results are reassembled."""
    context = ReportEvaluatorContext(
        name=report.name,
        report=report,
        experiment_metadata=report.experiment_metadata,
    )
    for evaluator in dataset.report_evaluators:
        evaluator_name = evaluator.get_serialization_name()
        try:
            result = await evaluator.evaluate_async(context)
        except Exception as error:  # noqa: BLE001
            report.report_evaluator_failures.append(
                EvaluatorFailure(
                    name=evaluator_name,
                    error_message=f"{type(error).__name__}: {error}",
                    error_stacktrace=traceback.format_exc(),
                    source=evaluator.as_spec(),
                    error_type=type(error).__name__,
                )
            )
        else:
            if isinstance(result, list):
                report.analyses.extend(result)
            else:
                report.analyses.append(result)


def _experiment_metadata(run_name: str, run_config: BenchmarkRunConfig) -> dict:
    return {
        "run_config": run_name,
        "agent": run_config.agent.experiment_metadata(),
        "benchmark": run_config.model_dump(mode="json", exclude={"agent"}),
    }


def evaluate(
    dataset: BenchmarkDataset,
    task: AgentTask,
    run_name: str,
    run_config: BenchmarkRunConfig,
    artifacts_dir: Path,
    checkpoint_path: Path,
    redo_tasks: Collection[str] = (),
    artifact_source_dir: Path | None = None,
) -> BenchmarkReport:
    metadata = _experiment_metadata(run_name, run_config)
    checkpoint = EvaluationCheckpoint(
        checkpoint_path,
        name=run_name,
        experiment_metadata=metadata,
    )
    _validate_redo_tasks(dataset, redo_tasks)
    if artifact_source_dir is not None:
        checkpoint.relocate_artifacts(artifact_source_dir, artifacts_dir)

    redo_attempt_names = {
        attempt_name
        for task_name in redo_tasks
        for attempt_name in _task_attempt_names(task_name, run_config.repetitions)
    }
    if redo_attempt_names:
        invalidated = checkpoint.invalidate(redo_attempt_names)
        _remove_invalidated_artifacts(invalidated, artifacts_dir)
        print(
            f"Redoing {len(redo_attempt_names)} attempt(s) from "
            f"{len(set(redo_tasks))} task(s)."
        )

    pending_dataset, attempts, attempt_names = _expand_attempts(
        dataset,
        run_config.repetitions,
        checkpoint.completed_names,
    )
    completed_count = len(attempt_names) - len(pending_dataset.cases)
    if completed_count:
        print(
            f"Resuming benchmark: {completed_count}/{len(attempt_names)} "
            "attempts already complete."
        )

    partial_report: BenchmarkReport | None = None
    if pending_dataset.cases:
        partial_report = pending_dataset.evaluate_sync(
            # Evals detects coroutine functions, but not async callable objects.
            # Keep model requests on the cancellable evaluation event loop.
            task.__call__ if isinstance(task, AgentTask) else task,
            name=run_name,
            max_concurrency=run_config.max_concurrency,
            retry_task=RetryConfig(
                stop=stop_after_attempt(run_config.task_max_attempts),
                wait=wait_exponential(multiplier=15, min=15, max=60),
                retry=retry_if_exception_type(ModelAPIError),
                reraise=True,
            ),
            lifecycle=LifecycleFactory(
                artifacts_dir,
                checkpoint,
                attempts,
            ),
            metadata=metadata,
        )

    report = checkpoint.completed_report(
        attempt_names,
        trace_id=partial_report.trace_id if partial_report else None,
        span_id=partial_report.span_id if partial_report else None,
    )
    asyncio.run(_apply_report_evaluators(report, dataset))
    return report


def run_benchmark(
    run_name: str,
    run_config: BenchmarkRunConfig,
    resume_directory: Path | None = None,
    redo_tasks: Collection[str] = (),
    progress_log: Path | None = None,
) -> Path:
    if redo_tasks and resume_directory is None:
        raise ValueError("redo_tasks requires resume_directory")
    source_directory = resume_directory.resolve() if resume_directory else None
    if source_directory is not None:
        if not source_directory.is_dir():
            raise FileNotFoundError(
                f"Resume directory does not exist: {source_directory}"
            )
        if not (source_directory / "artifacts").is_dir():
            raise ValueError(
                f"Resume directory has no artifacts folder: {source_directory}"
            )

    dataset = load_dataset(run_config.dataset)
    _validate_redo_tasks(dataset, redo_tasks)
    if source_directory is not None:
        source_checkpoint = source_directory / "checkpoint.json"
        if not source_checkpoint.is_file():
            raise ValueError(f"Resume source has no checkpoint: {source_checkpoint}")
        # Reject incompatible or corrupt checkpoints before copying files or
        # initializing database/model dependencies.
        EvaluationCheckpoint(
            source_checkpoint,
            name=run_name,
            experiment_metadata=_experiment_metadata(run_name, run_config),
        )
    artifact_source_dir: Path | None = None
    if redo_tasks:
        assert source_directory is not None
        run_directory = create_run_directory(run_name, run_config.dataset)
        _copy_run_for_redo(source_directory, run_directory)
        artifact_source_dir = source_directory / "artifacts"
        print(f"Copied benchmark run from: {source_directory}")
    elif source_directory is not None:
        run_directory = source_directory
    else:
        run_directory = create_run_directory(run_name, run_config.dataset)

    with (
        ProgressLog(
            progress_log,
            run_name,
            run_directory,
            len(dataset.cases) * run_config.repetitions,
        )
        if progress_log is not None
        else nullcontext()
    ):
        print(f"Benchmark run directory: {run_directory}")
        configure_observability(run_config.agent)

        task = AgentTask(
            agent=create_agent(run_config.agent.enabled_capabilities),
            deps=create_dependencies(run_config.agent),
            model=run_config.agent.create_model(),
            model_settings=(
                ModelSettings(temperature=run_config.agent.temperature)
                if run_config.agent.temperature is not None
                else ModelSettings()
            ),
            usage_limits=UsageLimits(request_limit=run_config.agent.request_limit),
        )
        report = evaluate(
            dataset,
            task,
            run_name,
            run_config,
            run_directory / "artifacts",
            run_directory / "checkpoint.json",
            redo_tasks=redo_tasks,
            artifact_source_dir=artifact_source_dir,
        )

        report.print(
            include_input=True,
            include_output=True,
            include_expected_output=True,
            include_durations=False,
        )
        result_path = save_report(report, run_directory / "report.json")
        print(f"\nSaved benchmark results to {result_path}")
        return result_path
