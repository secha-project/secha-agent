from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import uuid4

from pyspark.sql import SparkSession

if TYPE_CHECKING:
    from .config import AgentConfig


class SparkActionTimeoutError(TimeoutError):
    """One tagged Spark action exceeded its deadline."""


async def run_spark_with_timeout[T](
    spark: SparkSession,
    action: Callable[[], T],
    *,
    seconds: float = 60,
) -> T:
    if seconds <= 0:
        raise ValueError("seconds must be greater than zero")

    tag = f"query-{uuid4().hex}"

    def tagged_action() -> T:
        spark.addTag(tag)
        try:
            return action()
        finally:
            spark.removeTag(tag)

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(tagged_action),
            timeout=seconds,
        )
    except TimeoutError as exc:
        await asyncio.to_thread(spark.interruptTag, tag)
        raise SparkActionTimeoutError(
            f"Spark action timed out after {seconds:g} seconds."
        ) from exc


def get_spark_session(config: AgentConfig) -> SparkSession:
    spark: SparkSession = (
        SparkSession.builder.config(
            f"spark.sql.catalog.{config.catalog_name}",
            "io.unitycatalog.spark.UCSingleCatalog",
        )
        .config(
            f"spark.sql.catalog.{config.catalog_name}.uri",
            config.catalog_url,
        )
        .config(
            f"spark.sql.catalog.{config.catalog_name}.token",
            config.catalog_token,
        )
        .config("spark.sql.defaultCatalog", config.catalog_name)
        .remote(config.spark_url)
        .getOrCreate()
    )
    spark.catalog.setCurrentDatabase(config.schema_name)

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
