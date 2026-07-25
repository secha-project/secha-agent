import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import logfire
from pydantic_ai import ModelSettings, UsageLimits
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models import Model
from pydantic_ai.retries import RetryConfig
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

from src.agent.agent import Deps, agent
from src.agent.catalog import UnityCatalogClient
from src.agent.config import AgentConfig, get_model
from src.agent.spark_utils import get_spark_session
from src.validation.config import BENCHMARKS_DIR, RESULTS_DIR, BenchmarkRunConfig
from src.validation.eval_models import (
    BenchmarkDataset,
    NumberOutput,
    TableOutput,
    TextOutput,
)
from src.validation.lifecycle import LifecycleFactory
from src.validation.number_validation import ValidateNumberOutput
from src.validation.report_evaluators import DEFAULT_REPORT_EVALUATORS
from src.validation.reporting import BenchmarkReport, save_report
from src.validation.table_validation import ValidateTableIncluded, ValidateTableValues
from src.validation.text_validation import ValidateTextOutput

CUSTOM_EVALUATOR_TYPES = (
    ValidateTextOutput,
    ValidateNumberOutput,
    ValidateTableIncluded,
    ValidateTableValues,
)


@dataclass(frozen=True)
class AgentTask:
    deps: Deps
    model: Model
    model_settings: ModelSettings
    usage_limits: UsageLimits

    def __call__(self, prompt: str) -> str:
        return agent.run_sync(
            prompt,
            deps=self.deps,
            model=self.model,
            model_settings=self.model_settings,
            usage_limits=self.usage_limits,
        ).output


def configure_observability(config: AgentConfig) -> None:
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = config.otel_exporter_otlp_endpoint
    logfire.configure(
        send_to_logfire=False,
        service_name=config.logfire_service_name,
        metrics=False,
    )
    logfire.instrument_pydantic_ai()


def create_dependencies(config: AgentConfig) -> Deps:
    return Deps(
        spark=get_spark_session(config),
        catalog=UnityCatalogClient(
            config.catalog_url,
            config.catalog_token,
            config.catalog_name,
        ),
        schema_name=config.schema_name,
        table_column_page_size=config.table_column_page_size,
        evaluation_mode=True,
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


def create_run_directory(run_name: str, dataset_name: str) -> Path:
    dataset_directory = RESULTS_DIR / dataset_name
    dataset_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_directory = dataset_directory / f"{run_name}_{timestamp}"
    run_directory.mkdir()
    (run_directory / "artifacts").mkdir()
    return run_directory


def evaluate(
    dataset: BenchmarkDataset,
    task: AgentTask,
    run_name: str,
    run_config: BenchmarkRunConfig,
    artifacts_dir: Path,
) -> BenchmarkReport:
    return dataset.evaluate_sync(
        task,
        name=run_name,
        max_concurrency=run_config.max_concurrency,
        repeat=run_config.repetitions,
        retry_task=RetryConfig(
            stop=stop_after_attempt(run_config.task_max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=2),
            retry=retry_if_exception_type(ModelAPIError),
            reraise=True,
        ),
        lifecycle=LifecycleFactory(artifacts_dir, run_config.repetitions),
        metadata={
            "run_config": run_name,
            "agent": run_config.agent.experiment_metadata(),
            "benchmark": run_config.model_dump(mode="json", exclude={"agent"}),
        },
    )


def run_benchmark(run_name: str, run_config: BenchmarkRunConfig) -> Path:
    run_directory = create_run_directory(run_name, run_config.dataset)
    configure_observability(run_config.agent)

    task = AgentTask(
        deps=create_dependencies(run_config.agent),
        model=get_model(run_config.agent),
        model_settings=ModelSettings(temperature=run_config.agent.temperature),
        usage_limits=UsageLimits(request_limit=run_config.agent.request_limit),
    )
    dataset = load_dataset(run_config.dataset)
    report = evaluate(
        dataset,
        task,
        run_name,
        run_config,
        run_directory / "artifacts",
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
