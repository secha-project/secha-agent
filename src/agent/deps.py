from dataclasses import dataclass, field

from pyspark.sql import SparkSession

from .artifacts.base import ArtifactContext
from .catalog import UnityCatalogClient


@dataclass
class Deps:
    spark: SparkSession
    catalog: UnityCatalogClient
    schema_name: str
    table_column_page_size: int
    artifact_context: ArtifactContext = field(default_factory=ArtifactContext)
