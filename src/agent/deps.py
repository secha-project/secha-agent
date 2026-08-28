from dataclasses import dataclass, field

from pyspark.sql import SparkSession

from .catalog import UnityCatalogClient
from .outputs.artifacts import ArtifactCollector
from .outputs.tables import TableOutputStore
from .semantic_runtime import SemanticRuntime


@dataclass
class Deps:
    spark: SparkSession
    catalog: UnityCatalogClient
    schema_name: str
    table_column_page_size: int
    semantic: SemanticRuntime | None = None
    table_outputs: TableOutputStore = field(default_factory=TableOutputStore)
    artifacts: ArtifactCollector = field(default_factory=ArtifactCollector)
    table_artifact_max_rows: int | None = 10_000
