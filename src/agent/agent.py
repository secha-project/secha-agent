import logging
from dataclasses import dataclass, field
from pathlib import Path

import logfire
from pydantic_ai import Agent, ModelRetry, RunContext, capture_run_messages
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_settings import BaseSettings, SettingsConfigDict
from pyspark.sql import DataFrame, SparkSession


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

logfire.configure(
    send_to_logfire=False,
    console=logfire.ConsoleOptions(
        verbose=True,
        show_project_link=False,
    ),
)
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()


SCHEMA_NAME = "bronze"
SCHEMA_COLUMN_SAMPLE_SIZE = 8
TABLE_COLUMN_PAGE_SIZE = 20
MODEL_PROVIDER = "openai"  # "tuni" or "openai"
OPENAI_MODEL_NAME = "gpt-5.4-mini"


class AppConfig(BaseSettings):
    catalog_url: str
    catalog_token: str
    catalog_name: str = "analytics"
    spark_url: str
    tuni_inference_url: str
    tuni_inference_token: str
    tuni_inference_model: str
    openai_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env"
    )


def get_spark_session(app_config: AppConfig) -> SparkSession:
    logger.info("Connecting to Spark")
    spark: SparkSession = (
        SparkSession.builder.config(
            f"spark.sql.catalog.{app_config.catalog_name}",
            "io.unitycatalog.spark.UCSingleCatalog",
        )
        .config(
            f"spark.sql.catalog.{app_config.catalog_name}.uri",
            app_config.catalog_url,
        )
        .config(
            f"spark.sql.catalog.{app_config.catalog_name}.token",
            app_config.catalog_token,
        )
        .config("spark.sql.defaultCatalog", app_config.catalog_name)
        .remote(app_config.spark_url)
        .getOrCreate()
    )

    logger.info("Spark session established")
    return spark


def get_model(app_config: AppConfig) -> OpenAIChatModel:
    if MODEL_PROVIDER == "tuni":
        model_name = app_config.tuni_inference_model
        provider = OpenAIProvider(
            base_url=app_config.tuni_inference_url,
            api_key=app_config.tuni_inference_token,
        )
    elif MODEL_PROVIDER == "openai":
        model_name = OPENAI_MODEL_NAME
        provider = OpenAIProvider(api_key=app_config.openai_api_key)
    else:
        raise ValueError(
            f"Unknown MODEL_PROVIDER {MODEL_PROVIDER!r}; use 'tuni' or 'openai'."
        )

    logger.info("Configuring %s model %s", MODEL_PROVIDER, model_name)
    return OpenAIChatModel(model_name, provider=provider)


@dataclass
class Deps:
    spark: SparkSession
    outputs: dict[str, DataFrame] = field(default_factory=dict)

    def store_dataframe(self, value: DataFrame) -> str:
        """Store lazy spark dataframe in deps and return the reference such as Out[1] to be used by the LLM."""
        ref = f"Out[{len(self.outputs) + 1}]"
        self.outputs[ref] = value
        return ref

    def get_dataframe(self, ref: str) -> DataFrame:
        """Get a lazy spark dataframe from deps by its reference."""
        if ref not in self.outputs:
            raise ModelRetry(
                f"Error: {ref} is not a valid variable reference. Check the previous messages and try again."
            )
        return self.outputs[ref]


agent = Agent(
    deps_type=Deps,
    name="secha_agent",
    instructions=(
        "You are a data analytics assistant. "
        "Use tools only when information is missing. "
        "As soon as a tool result answers the user's question, stop calling tools "
        "and provide a concise final answer. Never repeat an identical tool call. "
        "Do not inspect unrelated columns."
    ),
)


@agent.tool
def describe_schema(ctx: RunContext[Deps]) -> str:
    """Summarize the tables and a sample of their columns in the configured schema."""
    schema = SCHEMA_NAME.split(".")[-1]
    quoted_schema = ".".join(f"`{part}`" for part in SCHEMA_NAME.split("."))
    tables = ctx.deps.spark.sql(
        f"SHOW TABLE EXTENDED IN {quoted_schema} LIKE '*'"
    ).collect()
    sections = []
    for table_number, table in enumerate(tables, start=1):
        schema_lines = table.information.split("Schema: root", 1)[1].splitlines()
        columns = [
            line.removeprefix(" |-- ")
            for line in schema_lines
            if line.startswith(" |-- ")
        ]
        column_sample = columns[:SCHEMA_COLUMN_SAMPLE_SIZE]
        sample = ", ".join(
            f"{column_name} ({column_details.rsplit(' (nullable = ', 1)[0]})"
            for column_name, column_details in (
                column.split(": ", 1) for column in column_sample
            )
        )
        sections.append(
            f"## Table {table_number}\n\n"
            f"Table: {schema}.{table.tableName}  \n"
            f"Columns Shown: {len(column_sample)} of {len(columns)}  \n"
            f"Column Sample: {sample}"
        )

    return "\n\n".join(sections) or f"No tables found in schema {SCHEMA_NAME}."


@agent.tool
def describe_table(
    ctx: RunContext[Deps],
    table_name: str,
    start_column: int = 1,
    end_column: int | None = None,
    column_names: list[str] | None = None,
) -> str:
    """Show column names, types, and comments for a table.

    By default this returns the first ``TABLE_COLUMN_PAGE_SIZE`` columns. Use
    ``start_column`` and ``end_column`` for a different 1-based inclusive range,
    or ``column_names`` to request particular columns regardless of their
    positions. ``column_names`` takes precedence over the range.

    Args:
        table_name: A schema-qualified table name such as ``bronze.events``.
        start_column: First column position to show (1-based and inclusive).
        end_column: Last column position to show (1-based and inclusive).
        column_names: Specific column names whose metadata and comments to show.
    """
    quoted_table_name = ".".join(f"`{part}`" for part in table_name.split("."))
    rows = ctx.deps.spark.sql(f"DESCRIBE TABLE {quoted_table_name}").collect()

    columns = []
    for position, row in enumerate(rows, start=1):
        if not row.col_name or row.col_name.startswith("#"):
            break
        columns.append((position, row))
    if column_names:
        requested_names = set(column_names)
        selected = [
            column for column in columns if column[1].col_name in requested_names
        ]
    else:
        last_column = end_column or start_column + TABLE_COLUMN_PAGE_SIZE - 1
        selected = columns[start_column - 1 : last_column]

    sections = []
    for position, column in selected:
        sections.append(
            f"## Column {position}\n\n"
            f"col_name: {column.col_name}\n"
            f"data_type: {column.data_type}\n"
            f"comment: {column.comment or 'No comment'}"
        )

    total_columns = len(columns)
    if column_names:
        footer = (
            f"[showing {len(selected)} requested columns out of "
            f"{total_columns} columns]"
        )
    elif len(selected) == total_columns:
        footer = f"[showing all {total_columns} columns]"
    elif start_column == 1:
        footer = f"[only showing top {len(selected)} out of {total_columns} columns]"
    else:
        footer = (
            f"[showing columns {start_column}-{start_column + len(selected) - 1} "
            f"out of {total_columns} columns]"
        )

    return "\n\n".join([*sections, footer])


def main() -> None:
    logger.info("Starting SECHA agent")
    try:
        config = AppConfig()  # type: ignore[call-arg]
        deps = Deps(spark=get_spark_session(config))
        model = get_model(config)
        logger.info("Running agent")
        with capture_run_messages() as messages:
            try:
                result = agent.run_sync(
                    "You have access to a data lake, use the tools to explore schema and "
                    "tables to find a column called qfryzevar and what it means",
                    deps=deps,
                    model=model,
                )
            finally:
                for message in messages:
                    if isinstance(message, ModelRequest):
                        for part in message.parts:
                            if isinstance(part, ToolReturnPart):
                                logger.info(
                                    "Tool %s returned:\n%s",
                                    part.tool_name,
                                    part.content,
                                )
        logger.info("Agent run completed")
        print(result.output)
    except Exception:
        logger.exception("Agent execution failed")
        raise


if __name__ == "__main__":
    main()
