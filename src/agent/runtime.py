from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import uuid4

from pyspark.sql import SparkSession


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
