import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import quote
from urllib.request import Request, urlopen

import logfire
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_settings import BaseSettings, SettingsConfigDict
from pyspark.sql import DataFrame, SparkSession


class AppConfig(BaseSettings):
    catalog_url: str
    catalog_token: str
    catalog_name: str = "analytics"
    spark_url: str
    schema_name: str = "bronze"
    schema_column_sample_size: int = 8
    table_column_page_size: int = 20
    model_provider: Literal["tuni", "openai"] = "openai"
    tuni_inference_url: str
    tuni_inference_token: str
    tuni_inference_model: str
    openai_model_name: str = "gpt-5.4-mini"
    openai_api_key: str | None = None
    otel_exporter_otlp_endpoint: str = "http://localhost:6006"
    logfire_service_name: str = "secha-agent"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env"
    )


def get_spark_session(app_config: AppConfig) -> SparkSession:
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
    spark.catalog.setCurrentDatabase(app_config.schema_name)

    return spark


def get_model(app_config: AppConfig) -> OpenAIChatModel:
    if app_config.model_provider == "tuni":
        model_name = app_config.tuni_inference_model
        provider = OpenAIProvider(
            base_url=app_config.tuni_inference_url,
            api_key=app_config.tuni_inference_token,
        )
    elif app_config.model_provider == "openai":
        model_name = app_config.openai_model_name
        provider = OpenAIProvider(api_key=app_config.openai_api_key)
    else:
        raise ValueError(
            f"Unknown model provider {app_config.model_provider!r}; "
            "use 'tuni' or 'openai'."
        )

    return OpenAIChatModel(model_name, provider=provider)


class UnityCatalogClient:
    def __init__(self, url: str, token: str, catalog_name: str):
        self._url = url
        self._token = token
        self.name = catalog_name

    def get_table_description(self, full_table_name: str) -> str:
        """Fetch a table's description from the Unity Catalog REST API."""
        url = (
            f"{self._url.rstrip('/')}/api/2.1/unity-catalog/tables/"
            f"{quote(full_table_name, safe='.')}"
        )
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._token}",
            },
        )
        with urlopen(request, timeout=10) as response:
            table = json.load(response)
        return table.get("comment") or "No description"


@dataclass
class Deps:
    spark: SparkSession
    catalog: UnityCatalogClient
    schema_name: str
    schema_column_sample_size: int
    table_column_page_size: int
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


def table_to_markdown(
    section_name: str,
    field_names: list[str],
    rows: list[list[object]],
) -> str:
    """Render rows as Markdown sections, using each row's first value in its heading."""
    sections = []
    for section_number, *values in rows:
        fields = "\n".join(
            f"{field_name}: {value}" for field_name, value in zip(field_names, values)
        )
        sections.append(f"## {section_name} {section_number}\n\n{fields}")
    return "\n\n".join(sections)


def dataframe_to_markdown(
    dataframe: DataFrame,
    start_row: int = 1,
    end_row: int | None = None,
    column_names: list[str] | None = None,
) -> str:
    """Render a 1-based inclusive range of dataframe rows as Markdown sections."""
    end_row = end_row or start_row + 9
    available_columns = dataframe.schema.fieldNames()
    selected_columns = column_names or available_columns[:10]

    unknown_columns = [
        column_name
        for column_name in selected_columns
        if column_name not in available_columns
    ]
    if unknown_columns:
        raise ModelRetry(
            f"Unknown dataframe columns: {', '.join(unknown_columns)}. "
            f"Available columns: {', '.join(available_columns)}."
        )

    rows = (
        dataframe.select(*selected_columns)
        .limit(end_row)
        .collect()[start_row - 1 : end_row]
    )
    table_rows = [
        [row_number, *row] for row_number, row in enumerate(rows, start=start_row)
    ]
    table = table_to_markdown("Row", selected_columns, table_rows)

    if column_names:
        column_summary = f"{len(selected_columns)} requested columns"
    elif len(selected_columns) == len(available_columns):
        column_summary = f"all {len(available_columns)} columns"
    else:
        column_summary = (
            f"top {len(selected_columns)} out of {len(available_columns)} columns"
        )
    footer = (
        f"[showing {len(rows)} rows from requested range "
        f"{start_row}-{end_row}, {column_summary}]"
    )
    return "\n\n".join([table, footer]) if table else footer


def resolve_table_name(table_name: str, schema_name: str) -> str:
    """Qualify a bare table name with the configured schema."""
    if "." not in table_name:
        return f"{schema_name}.{table_name}"
    return table_name


@agent.tool
def describe_schema(ctx: RunContext[Deps]) -> str:
    """Summarize the tables and a sample of their columns in the configured schema."""
    schema_name = ctx.deps.schema_name
    schema = schema_name.split(".")[-1]
    quoted_schema = ".".join(f"`{part}`" for part in schema_name.split("."))
    tables = ctx.deps.spark.sql(
        f"SHOW TABLE EXTENDED IN {quoted_schema} LIKE '*'"
    ).collect()
    table_rows = []
    for table_number, table in enumerate(tables, start=1):
        information_lines = table.information.splitlines()
        description = next(
            (
                line.removeprefix("Comment: ")
                for line in information_lines
                if line.startswith("Comment: ")
            ),
            "No description",
        )
        schema_lines = table.information.split("Schema: root", 1)[1].splitlines()
        columns = [
            line.removeprefix(" |-- ")
            for line in schema_lines
            if line.startswith(" |-- ")
        ]
        column_sample = columns[: ctx.deps.schema_column_sample_size]
        sample = ", ".join(
            f"{column_name} ({column_details.rsplit(' (nullable = ', 1)[0]})"
            for column_name, column_details in (
                column.split(": ", 1) for column in column_sample
            )
        )
        table_rows.append(
            [
                table_number,
                f"{schema}.{table.tableName}",
                description,
                f"{len(column_sample)} of {len(columns)}",
                sample,
            ]
        )

    if not table_rows:
        return f"No tables found in schema {schema_name}."

    return table_to_markdown(
        "Table",
        ["Table", "Description", "Columns Shown", "Column Sample"],
        table_rows,
    )


@agent.tool
def describe_table(
    ctx: RunContext[Deps],
    table_name: str,
    start_column: int = 1,
    end_column: int | None = None,
    column_names: list[str] | None = None,
) -> str:
    """Show column names, types, and comments for a table.

    By default this returns the configured number of columns. Use ``start_column``
    and ``end_column`` for a different 1-based inclusive range, or
    ``column_names`` to request particular columns regardless of their positions.
    ``column_names`` takes precedence over the range.

    Args:
        table_name: A bare table name, or a schema-qualified name such as
            ``bronze.events``. Bare names use the configured schema.
        start_column: First column position to show (1-based and inclusive).
        end_column: Last column position to show (1-based and inclusive).
        column_names: Specific column names whose metadata and comments to show.
    """
    resolved_table_name = resolve_table_name(table_name, ctx.deps.schema_name)
    full_table_name = (
        resolved_table_name
        if resolved_table_name.count(".") == 2
        else f"{ctx.deps.catalog.name}.{resolved_table_name}"
    )
    table_description = ctx.deps.catalog.get_table_description(full_table_name)
    quoted_table_name = ".".join(
        f"`{part}`" for part in resolved_table_name.split(".")
    )
    metadata = ctx.deps.spark.sql(f"DESCRIBE TABLE {quoted_table_name}").collect()

    columns = []
    for position, row in enumerate(metadata, start=1):
        if not row.col_name or row.col_name.startswith("#"):
            break
        columns.append((position, row))
    if column_names:
        requested_names = set(column_names)
        selected = [
            column for column in columns if column[1].col_name in requested_names
        ]
    else:
        last_column = end_column or start_column + ctx.deps.table_column_page_size - 1
        selected = columns[start_column - 1 : last_column]

    table_rows = [
        [position, column.col_name, column.data_type, column.comment or "No comment"]
        for position, column in selected
    ]

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

    table = table_to_markdown(
        "Column", ["col_name", "data_type", "comment"], table_rows
    )
    summary = f"Table: {full_table_name}\nDescription: {table_description}"
    return "\n\n".join(part for part in [summary, table, footer] if part)


def concise_spark_error(exc: Exception) -> str:
    """Return Spark's useful error details without the noisy JVM stack trace."""
    return str(exc).partition("JVM stacktrace:")[0].rstrip()


@agent.tool
def execute_sql(ctx: RunContext[Deps], sql: str) -> str:
    """Run SQL, save its lazy dataframe, and preview its first 10 rows and columns.

    Spark validates the query by resolving its output schema before any result is
    stored. Invalid SQL is returned for correction and retry.

    Args:
        sql: The Spark SQL query to execute.
    """
    if not sql.strip():
        raise ModelRetry("SQL validation failed (EmptySql): SQL text is empty.")

    try:
        result = ctx.deps.spark.sql(sql)
        result.schema
    except Exception as exc:
        get_error_class = getattr(exc, "getErrorClass", None)
        error_type = (
            get_error_class() if callable(get_error_class) else type(exc).__name__
        )
        raise ModelRetry(
            f"SQL validation failed ({error_type}): {concise_spark_error(exc)}. "
        ) from exc

    try:
        preview = dataframe_to_markdown(result)
    except ModelRetry:
        raise
    except Exception as exc:
        get_error_class = getattr(exc, "getErrorClass", None)
        error_type = (
            get_error_class() if callable(get_error_class) else type(exc).__name__
        )
        raise ModelRetry(
            "SQL execution failed while reading the preview "
            f"({error_type}): {concise_spark_error(exc)}. "
        ) from exc

    ref = ctx.deps.store_dataframe(result)
    return f"DataFrame: {ref}\n\n{preview}"


@agent.tool
def inspect_dataframe(
    ctx: RunContext[Deps],
    ref: str,
    start_row: int = 1,
    end_row: int | None = None,
    column_names: list[str] | None = None,
) -> str:
    """Inspect rows and columns from a dataframe saved by ``execute_sql``.

    The row range is 1-based and inclusive. By default, this shows 10 rows and
    the first 10 columns. ``column_names`` selects specific columns instead.
    Row positions are only deterministic when the original SQL defines an order.

    Args:
        ref: A saved dataframe tag such as ``Out[1]``.
        start_row: First row to show (1-based and inclusive).
        end_row: Last row to show (1-based and inclusive).
        column_names: Specific columns to show instead of the first 10.
    """
    dataframe = ctx.deps.get_dataframe(ref)
    preview = dataframe_to_markdown(
        dataframe,
        start_row=start_row,
        end_row=end_row,
        column_names=column_names,
    )
    return f"DataFrame: {ref}\n\n{preview}"


def main() -> None:
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
        schema_column_sample_size=config.schema_column_sample_size,
        table_column_page_size=config.table_column_page_size,
    )
    model = get_model(config)
    result = agent.run_sync(
        "What is the reporting interval in seconds of the data in "
        "power_quality_measurements_electrix?",
        deps=deps,
        model=model,
    )
    print(result.output)


if __name__ == "__main__":
    main()
