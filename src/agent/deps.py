from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession

from src.app.dataframes import DataFrameReferences

from .catalog import UnityCatalogClient

DataframeExporter = Callable[[SparkSession, DataFrame], Awaitable[None]]


@dataclass
class Deps:
    spark: SparkSession
    catalog: UnityCatalogClient
    schema_name: str
    table_column_page_size: int
    dataframe_references: DataFrameReferences
    dataframe_exporter: DataframeExporter | None = None
