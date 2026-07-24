from pydantic_ai import ModelRetry
from pyspark.sql import DataFrame, SparkSession

from .config import AgentConfig


def get_spark_session(app_config: AgentConfig) -> SparkSession:
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
    range_was_requested = end_row is not None
    end_row = end_row or start_row + 9
    requested_row_count = end_row - start_row + 1
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

    total_rows = dataframe.count()
    collected_rows = dataframe.select(*selected_columns).limit(end_row).collect()
    rows = collected_rows[start_row - 1 : end_row]
    has_more_rows = total_rows > requested_row_count
    has_more_columns = len(selected_columns) < len(available_columns)
    table_rows = [
        [row_number, *row] for row_number, row in enumerate(rows, start=start_row)
    ]
    table = table_to_markdown("Row", selected_columns, table_rows)

    overflow_messages = []
    if has_more_rows:
        row_selection = "requested" if range_was_requested else "default"
        row_label = "row" if len(rows) == 1 else "rows"
        overflow_messages.append(
            f"showing only the {row_selection} {len(rows)} {row_label} "
            f"out of the {total_rows} available"
        )
    if has_more_columns:
        column_selection = "requested" if column_names else "default"
        column_label = "column" if len(selected_columns) == 1 else "columns"
        overflow_messages.append(
            f"showing only the {column_selection} {len(selected_columns)} "
            f"{column_label} "
            f"out of the {len(available_columns)} available"
        )

    if not overflow_messages:
        return table

    footer = f"[{', '.join(overflow_messages)}]"
    return "\n\n".join([table, footer]) if table else footer


def concise_spark_error(exc: Exception) -> str:
    """Return Spark's useful error details without the noisy JVM stack trace."""
    return str(exc).partition("JVM stacktrace:")[0].rstrip()


def spark_error_type(exc: Exception) -> str:
    """Return Spark's structured error class when one is available."""
    get_error_class = getattr(exc, "getErrorClass", None)
    if callable(get_error_class):
        error_class = get_error_class()
        if isinstance(error_class, str) and error_class:
            return error_class
    return type(exc).__name__
