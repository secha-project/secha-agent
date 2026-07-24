import argparse
import json
import os
from contextvars import Token
from datetime import datetime
from pathlib import Path

import logfire
from pydantic_ai import ModelSettings, UsageLimits
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.retries import RetryConfig
from pydantic_evals.lifecycle import CaseLifecycle
from pydantic_evals.reporting import EvaluationReport, EvaluationReportAdapter
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

from src.agent.agent import (
    Deps,
    agent,
    reset_evaluation_csv_path,
    set_evaluation_csv_path,
)
from src.agent.catalog import UnityCatalogClient
from src.agent.config import AgentConfig, get_model
from src.agent.spark_utils import get_spark_session

from src.validation.config import (
    BENCHMARKS_DIR,
    RESULTS_DIR,
    BenchmarkRunConfig,
    load_run_config,
)
from src.validation.eval_models import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkOutput,
    CaseMetadata,
    NumberOutput,
    TableOutput,
    TextOutput,
)
from src.validation.number_validation import ValidateNumberOutput
from src.validation.report_evaluators import DEFAULT_REPORT_EVALUATORS
from src.validation.table_validation import (
    ValidateTableIncluded,
    ValidateTableValues,
)
from src.validation.text_validation import ValidateTextOutput


CUSTOM_EVALUATOR_TYPES = (
    ValidateTextOutput,
    ValidateNumberOutput,
    ValidateTableIncluded,
    ValidateTableValues,
)


class BenchmarkLifecycle(CaseLifecycle[str, BenchmarkOutput, CaseMetadata]):
    """Configure the persistent CSV artifact path for an evaluation case."""

    def __init__(
        self,
        case: BenchmarkCase,
        *,
        artifacts_dir: Path,
        repetition_index: int | None = None,
    ) -> None:
        super().__init__(case)
        self.artifacts_dir = artifacts_dir
        self.repetition_index = repetition_index
        self.csv_path_token: Token[Path | None] | None = None

    async def setup(self) -> None:
        task_name = self.case.name
        if task_name is None:
            raise ValueError("Evaluation cases must have names to store CSV artifacts.")

        artifact_name = (
            task_name
            if self.repetition_index is None
            else f"{task_name}_{self.repetition_index}"
        )
        filename = f"{artifact_name}.csv"
        if Path(filename).name != filename:
            raise ValueError(
                f"Evaluation case name {task_name!r} cannot be used as an artifact name."
            )
        self.csv_path_token = set_evaluation_csv_path(
            self.artifacts_dir / filename
        )

    async def teardown(self, result: object) -> None:
        if self.csv_path_token is not None:
            reset_evaluation_csv_path(self.csv_path_token)
            self.csv_path_token = None


def add_case_evaluators(dataset: BenchmarkDataset, artifact_base_path: Path) -> None:
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


def create_run_directory() -> Path:
    """Create the timestamped folder shared by a report and its artifacts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_directory = RESULTS_DIR / timestamp
    run_directory.mkdir()
    (run_directory / "artifacts").mkdir()
    return run_directory


def save_report(
    report: EvaluationReport[str, BenchmarkOutput, CaseMetadata],
    run_directory: Path,
) -> Path:
    """Save metadata first, with experiment trace identifiers nested inside it."""
    serialized_report = json.loads(EvaluationReportAdapter.dump_json(report))
    existing_metadata = serialized_report.pop("experiment_metadata") or {}
    trace_id = serialized_report.pop("trace_id")
    span_id = serialized_report.pop("span_id")
    serialized_report.pop("name")
    analyses = serialized_report.pop("analyses")
    cases = serialized_report.pop("cases")
    experiment_metadata = {
        "trace_id": trace_id,
        "span_id": span_id,
        **{
            key: value
            for key, value in existing_metadata.items()
            if key not in {"trace_id", "span_id"}
        },
    }
    report_document = {
        "experiment_metadata": experiment_metadata,
        "analyses": analyses,
        "cases": cases,
        **serialized_report,
    }

    result_path = run_directory / "report.json"
    result_path.write_text(
        json.dumps(report_document, indent=2) + "\n",
        encoding="utf-8",
    )
    return result_path


def run_benchmark(run_name: str, run_config: BenchmarkRunConfig) -> Path:
    agent_config = AgentConfig(**run_config.agent)  # type: ignore[call-arg]
    run_directory = create_run_directory()
    artifacts_dir = run_directory / "artifacts"

    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
        agent_config.otel_exporter_otlp_endpoint
    )
    logfire.configure(
        send_to_logfire=False,
        service_name=agent_config.logfire_service_name,
        metrics=False,
    )
    logfire.instrument_pydantic_ai()

    deps = Deps(
        spark=get_spark_session(agent_config),
        catalog=UnityCatalogClient(
            agent_config.catalog_url,
            agent_config.catalog_token,
            agent_config.catalog_name,
        ),
        schema_name=agent_config.schema_name,
        table_column_page_size=agent_config.table_column_page_size,
        evaluation_mode=True,
    )

    dataset_path = BENCHMARKS_DIR / run_config.dataset / "dataset.yaml"
    dataset = BenchmarkDataset.from_file(
        dataset_path,
        custom_evaluator_types=CUSTOM_EVALUATOR_TYPES,
    )
    add_case_evaluators(dataset, dataset_path.parent)
    dataset.report_evaluators.extend(
        evaluator_type() for evaluator_type in DEFAULT_REPORT_EVALUATORS
    )

    model = get_model(agent_config)

    def run_agent(prompt: str) -> str:
        return agent.run_sync(
            prompt,
            deps=deps,
            model=model,
            model_settings=ModelSettings(temperature=agent_config.temperature),
            usage_limits=UsageLimits(request_limit=agent_config.request_limit),
        ).output

    repetition_counts: dict[str | None, int] = {}

    def create_lifecycle(case: BenchmarkCase) -> BenchmarkLifecycle:
        repetition_index = None
        if run_config.repetitions > 1:
            repetition_index = repetition_counts.get(case.name, 0) + 1
            repetition_counts[case.name] = repetition_index
        return BenchmarkLifecycle(
            case,
            artifacts_dir=artifacts_dir,
            repetition_index=repetition_index,
        )

    report = dataset.evaluate_sync(
        run_agent,
        name=run_name,
        max_concurrency=run_config.max_concurrency,
        repeat=run_config.repetitions,
        retry_task=RetryConfig(
            stop=stop_after_attempt(run_config.task_max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=2),
            retry=retry_if_exception_type(ModelAPIError),
            reraise=True,
        ),
        lifecycle=create_lifecycle,
        metadata={
            "run_config": run_name,
            "agent": agent_config.experiment_metadata(),
            "benchmark": run_config.model_dump(mode="json", exclude={"agent"}),
        },
    )

    report.print(
        include_input=True,
        include_output=True,
        include_expected_output=True,
        include_durations=False,
    )
    result_path = save_report(report, run_directory)
    print(f"\nSaved benchmark results to {result_path}")
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configured benchmark")
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Run config name from src/validation/run_configs "
            "(for example, 'openai-mini')"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name, run_config = load_run_config(args.config)
    run_benchmark(run_name, run_config)


if __name__ == "__main__":
    main()
