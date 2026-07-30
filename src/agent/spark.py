from pyspark.sql import SparkSession

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
