from dataclasses import dataclass, field
from urllib.error import HTTPError

from pydantic_ai import Agent, ModelRetry, RunContext
from pyspark.sql import DataFrame, SparkSession

from .catalog import (
    BrowseTablesResult,
    ColumnMatch,
    DescribeTableResult,
    SearchColumnsResult,
    TableSummary,
    UnityCatalogClient,
    get_columns,
    normalize_search_text,
    truncate_description,
)
from .spark_utils import (
    concise_spark_error,
    dataframe_to_markdown,
    spark_error_type,
)


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


def full_table_name(table_name: str, deps: Deps) -> str:
    """Fully qualify a bare or schema-qualified table name."""
    resolved_table_name = (
        f"{deps.schema_name}.{table_name}" if "." not in table_name else table_name
    )
    if resolved_table_name.count(".") == 2:
        return resolved_table_name
    return f"{deps.catalog.name}.{resolved_table_name}"


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
            "Call browse_tables to find its canonical name."
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
            "Call browse_tables to find its canonical name."
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
