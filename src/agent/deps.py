from dataclasses import dataclass, field

from pyspark.sql import SparkSession

from .catalog import UnityCatalogClient
from .outputs.artifacts import ArtifactCollector
from .outputs.charts import ChartOutputStore
from .outputs.tables import TableOutputStore


@dataclass
class Deps:
    spark: SparkSession
    catalog: UnityCatalogClient
    schema_name: str
    table_column_page_size: int
    semantic_layer_enabled: bool = False
    table_outputs: TableOutputStore = field(default_factory=TableOutputStore)
    artifacts: ArtifactCollector = field(default_factory=ArtifactCollector)
    table_artifact_max_rows: int | None = 10_000
    chart_outputs: ChartOutputStore = field(default_factory=ChartOutputStore)
