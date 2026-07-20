from pydantic_ai import ModelRetry
from pyspark.sql import DataFrame, SparkSession

from .config import AppConfig


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


def concise_spark_error(exc: Exception) -> str:
    """Return Spark's useful error details without the noisy JVM stack trace."""
    return str(exc).partition("JVM stacktrace:")[0].rstrip()


def spark_error_type(exc: Exception) -> str:
    """Return Spark's structured error class when one is available."""
    get_error_class = getattr(exc, "getErrorClass", None)
    if callable(get_error_class):
        return get_error_class() or type(exc).__name__
    return type(exc).__name__
