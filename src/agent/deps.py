from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession

from .catalog import UnityCatalogClient
from .query_results import QueryResultRegistry

ResultExporter = Callable[[SparkSession, DataFrame], Awaitable[None]]


@dataclass
class Deps:
    spark: SparkSession
    catalog: UnityCatalogClient
    schema_name: str
    table_column_page_size: int
    result_registry: QueryResultRegistry
    result_exporter: ResultExporter | None = None
