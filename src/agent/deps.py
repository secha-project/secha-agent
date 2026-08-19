from dataclasses import dataclass, field

from pyspark.sql import SparkSession

from .artifacts.base import RunArtifacts
from .catalog import UnityCatalogClient
from .semantic_runtime import SemanticRuntime


@dataclass
class Deps:
    spark: SparkSession
    catalog: UnityCatalogClient
    schema_name: str
    table_column_page_size: int
    semantic: SemanticRuntime | None = None
    artifacts: RunArtifacts = field(default_factory=RunArtifacts)
