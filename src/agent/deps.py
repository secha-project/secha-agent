from dataclasses import dataclass, field

from pyspark.sql import SparkSession

from .artifacts.base import ArtifactContext
from .catalog import UnityCatalogClient
from .semantic import SemanticRuntime


@dataclass
class Deps:
    spark: SparkSession
    catalog: UnityCatalogClient
    schema_name: str
    table_column_page_size: int
    semantic: SemanticRuntime | None = None
    artifact_context: ArtifactContext = field(default_factory=ArtifactContext)
