import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import logfire
from pydantic import BaseModel
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


class TableSummary(BaseModel):
    name: str
    description: str | None


class BrowseTablesResult(BaseModel):
    schema_name: str
    tables: list[TableSummary]


class ColumnMetadata(BaseModel):
    position: int
    name: str
    data_type: str
    comment: str | None


class ColumnMatch(ColumnMetadata):
    matched_in: list[Literal["name", "comment"]]


class SearchColumnsResult(BaseModel):
    table_name: str
    query: str
    searched_columns: int
    total_matches: int
    matches: list[ColumnMatch]


class DescribeTableResult(BaseModel):
    table_name: str
    description: str | None
    total_columns: int
    columns: list[ColumnMetadata]
    missing_columns: list[str]


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

    def _get(self, path: str, params: dict[str, str] | None = None) -> dict:
        url = f"{self._url.rstrip('/')}/api/2.1/unity-catalog{path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._token}",
            },
        )
        with urlopen(request, timeout=10) as response:
            return json.load(response)

    def list_tables(self, schema_name: str) -> list[dict]:
        """Fetch lightweight table metadata for every table in a schema."""
        params = {
            "catalog_name": self.name,
            "schema_name": schema_name,
            "omit_columns": "true",
            "omit_properties": "true",
        }
        tables = []
        while True:
            page = self._get("/tables", params)
            tables.extend(page.get("tables") or [])
            page_token = page.get("next_page_token")
            if not page_token:
                return tables
            params["page_token"] = page_token

    def get_table(self, full_table_name: str) -> dict:
        """Fetch a table and all of its column metadata."""
        return self._get(f"/tables/{quote(full_table_name, safe='.')}")


@dataclass
class Deps:
    spark: SparkSession
    catalog: UnityCatalogClient
    schema_name: str
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
        "You have access to a schema in a Spark catalog, which you can explore "
        "using tools to answer user questions. "
        "Table names may be uncertain, and users may refer to columns "
        "using wording from their comments rather than their exact identifiers. "
        "Browse tables to resolve table names, search columns to resolve descriptive "
        "wording, and describe tables only for known identifiers."
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


def full_table_name(table_name: str, deps: Deps) -> str:
    """Fully qualify a bare or schema-qualified table name."""
    resolved_table_name = (
        f"{deps.schema_name}.{table_name}" if "." not in table_name else table_name
    )
    if resolved_table_name.count(".") == 2:
        return resolved_table_name
    return f"{deps.catalog.name}.{resolved_table_name}"


def truncate_description(description: str | None, max_length: int = 200) -> str | None:
    if not description:
        return None
    description = " ".join(description.split())
    if len(description) <= max_length:
        return description
    return f"{description[: max_length - 1].rstrip()}…"


def normalize_search_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def get_columns(table: dict) -> list[ColumnMetadata]:
    return [
        ColumnMetadata(
            position=position,
            name=column["name"],
            data_type=column.get("type_text") or column.get("type_name") or "unknown",
            comment=column.get("comment"),
        )
        for position, column in enumerate(table.get("columns") or [], start=1)
    ]


@agent.tool
def browse_tables(ctx: RunContext[Deps]) -> BrowseTablesResult:
    """List tables in the configured schema with canonical names and short descriptions.

    Use this when a table name is unknown, ambiguous, or possibly misspelled.
    This tool does not return column metadata.
    """
    schema_name = ctx.deps.schema_name.split(".")[-1]
    tables = ctx.deps.catalog.list_tables(schema_name)
    return BrowseTablesResult(
        schema_name=f"{ctx.deps.catalog.name}.{schema_name}",
        tables=[
            TableSummary(
                name=f"{ctx.deps.catalog.name}.{schema_name}.{table['name']}",
                description=truncate_description(table.get("comment")),
            )
            for table in tables
        ],
    )


@agent.tool
def search_columns(
    ctx: RunContext[Deps],
    table_name: str,
    query: str,
) -> SearchColumnsResult:
    """Search every column name and comment in one known table for text.

    Use this when the user describes a column in human language or its exact
    identifier is unknown. The table name must be exact; use ``browse_tables``
    when it is uncertain. This searches metadata, not row values.

    Args:
        table_name: A bare, schema-qualified, or fully qualified table name.
            Bare names use the configured schema.
        query: Text to find in column names or comments.
    """
    normalized_query = normalize_search_text(query)
    if not normalized_query:
        raise ModelRetry("Column search query cannot be empty.")

    resolved_table_name = full_table_name(table_name, ctx.deps)
    try:
        table = ctx.deps.catalog.get_table(resolved_table_name)
    except HTTPError as exc:
        if exc.code != 404:
            raise
        raise ModelRetry(
            f"Table {resolved_table_name!r} was not found. "
            "Call browse_tables to find its canonical name, then retry."
        ) from exc

    columns = get_columns(table)
    ranked_matches = []
    for column in columns:
        normalized_name = normalize_search_text(column.name)
        normalized_comment = normalize_search_text(column.comment or "")
        matched_in = []
        if normalized_query in normalized_name:
            matched_in.append("name")
        if normalized_query in normalized_comment:
            matched_in.append("comment")
        if matched_in:
            if normalized_query == normalized_name:
                rank = 0
            elif "name" in matched_in:
                rank = 1
            else:
                rank = 2
            ranked_matches.append(
                (
                    rank,
                    ColumnMatch(**column.model_dump(), matched_in=matched_in),
                )
            )

    ranked_matches.sort(key=lambda match: (match[0], match[1].position))
    matches = [match for _, match in ranked_matches]
    return SearchColumnsResult(
        table_name=resolved_table_name,
        query=query,
        searched_columns=len(columns),
        total_matches=len(matches),
        matches=matches[: ctx.deps.table_column_page_size],
    )


@agent.tool
def describe_table(
    ctx: RunContext[Deps],
    table_name: str,
    start_column: int = 1,
    end_column: int | None = None,
    column_names: list[str] | None = None,
) -> DescribeTableResult:
    """Show the full description and selected column metadata for one known table.

    By default this returns the configured first page of columns. Use
    ``start_column`` and ``end_column`` for a 1-based inclusive range, or
    ``column_names`` for exact identifiers. ``column_names`` takes precedence
    over the range. Use ``search_columns`` for human-language wording.

    Args:
        table_name: A bare, schema-qualified, or fully qualified table name.
            Bare names use the configured schema.
        start_column: First column position to return, 1-based and inclusive.
        end_column: Last column position to return, 1-based and inclusive.
        column_names: Exact column identifiers to return.
    """
    resolved_table_name = full_table_name(table_name, ctx.deps)
    try:
        table = ctx.deps.catalog.get_table(resolved_table_name)
    except HTTPError as exc:
        if exc.code != 404:
            raise
        raise ModelRetry(
            f"Table {resolved_table_name!r} was not found. "
            "Call browse_tables to find its canonical name, then retry."
        ) from exc

    columns = get_columns(table)
    if column_names:
        columns_by_name = {column.name: column for column in columns}
        selected = [
            columns_by_name[name] for name in column_names if name in columns_by_name
        ]
        missing_columns = [name for name in column_names if name not in columns_by_name]
    else:
        last_column = end_column or start_column + ctx.deps.table_column_page_size - 1
        selected = columns[start_column - 1 : last_column]
        missing_columns = []

    return DescribeTableResult(
        table_name=resolved_table_name,
        description=table.get("comment"),
        total_columns=len(columns),
        columns=selected,
        missing_columns=missing_columns,
    )


def concise_spark_error(exc: Exception) -> str:
    """Return Spark's useful error details without the noisy JVM stack trace."""
    return str(exc).partition("JVM stacktrace:")[0].rstrip()


def spark_error_type(exc: Exception) -> str:
    """Return Spark's structured error class when one is available."""
    get_error_class = getattr(exc, "getErrorClass", None)
    if callable(get_error_class):
        return get_error_class() or type(exc).__name__
    return type(exc).__name__


@agent.tool(retries=5)
def query_data(ctx: RunContext[Deps], sql: str) -> str:
    """Run Spark SQL and preview its first 10 rows and columns.

    Use this for questions about row data, calculations, aggregates, or data
    values. Do not use it to search catalog metadata. The lazy dataframe is
    saved and Spark validates the query before the preview is read.

    Args:
        sql: The Spark SQL query to execute.
    """
    if not sql.strip():
        raise ModelRetry("SQL validation failed (EmptySql): SQL text is empty.")

    try:
        result = ctx.deps.spark.sql(sql)
        result.schema
    except Exception as exc:
        raise ModelRetry(
            f"SQL validation failed ({spark_error_type(exc)}): "
            f"{concise_spark_error(exc)}. "
        ) from exc

    try:
        preview = dataframe_to_markdown(result)
    except ModelRetry:
        raise
    except Exception as exc:
        raise ModelRetry(
            "SQL execution failed while reading the preview "
            f"({spark_error_type(exc)}): {concise_spark_error(exc)}. "
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
    """Inspect rows and columns from a dataframe saved by ``query_data``.

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
        table_column_page_size=config.table_column_page_size,
    )
    model = get_model(config)
    result = agent.run_sync(
        "What are the units of the import active energy counter column "
        "in power_quality_mesurements_electrix?",
        deps=deps,
        model=model,
    )
    print(result.output)


if __name__ == "__main__":
    main()
