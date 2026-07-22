import os
from pathlib import Path

import logfire

from src.agent.agent import Deps, agent
from src.agent.catalog import UnityCatalogClient
from src.agent.config import AppConfig, get_model
from src.agent.spark_utils import get_spark_session

from src.validation.eval_models import (
    BenchmarkDataset,
    TextOutput,
    NumberOutput,
    TableOutput,
)
from src.validation.text_validation import ValidateTextOutput
from src.validation.number_validation import ValidateNumberOutput
from src.validation.table_validation import ValidateTableIncluded, ValidateTableValues


CUSTOM_EVALUATOR_TYPES = (ValidateTextOutput, ValidateNumberOutput, ValidateTableIncluded, ValidateTableValues)


def run_benchmark() -> None:
    config = AppConfig()  # type: ignore[call-arg]
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = config.otel_exporter_otlp_endpoint
    logfire.configure(
        send_to_logfire=False,
        service_name=config.logfire_service_name,
        metrics=False,
    )
    logfire.instrument_pydantic_ai()

    deps = Deps(
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

    dataset = BenchmarkDataset.from_file(
        Path(__file__).parent
        / "benchmark"
        / "power-quality-monitoring"
        / "dataset.yaml",
    )

    # TODO: Add the right evaluators to each case type after loading, they should be
    # in the dataset file but are not because I hadn't designed them up until now
    # I will add them later so they are loaded automatically, but for now let's do it dynamically

    model = get_model(config)

    def run_agent(prompt: str) -> str:
        return agent.run_sync(prompt, deps=deps, model=model).output

    report = dataset.evaluate_sync(run_agent)

    report.print(
        include_input=True,
        include_output=True,
        include_expected_output=True,
        include_durations=False,
    )


if __name__ == "__main__":
    run_benchmark()
