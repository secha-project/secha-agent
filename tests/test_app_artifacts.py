import asyncio
import json
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pytest

from src.agent.artifacts.base import (
    DEFAULT_TABLE_ARTIFACT_MAX_ROWS,
    Artifact,
    ArtifactContext,
    ArtifactPolicy,
    ArtifactRef,
    ArtifactRegistry,
)
from src.agent.artifacts.table import SparkTableSource, TableContent
from src.agent.artifacts.visualization import VisualizationContent
from src.app.artifact_renderer import ArtifactRenderer


class TableContainer:
    def __init__(self) -> None:
        self.data: Any = None
        self.dataframe_kwargs: dict[str, Any] = {}
        self.download_button_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def dataframe(self, data: Any, **kwargs: Any) -> None:
        self.data = data
        self.dataframe_kwargs = kwargs

    def caption(self, _value: str) -> None:
        pass

    def download_button(self, *args: Any, **kwargs: Any) -> None:
        self.download_button_calls.append((args, kwargs))


class MaterializedResult:
    def __init__(self) -> None:
        self.limit_value: int | None = None

    def limit(self, value: int) -> "MaterializedResult":
        self.limit_value = value
        return self

    def toPandas(self) -> pd.DataFrame:
        return pd.DataFrame([{"id": 1}])


class TaggedSpark:
    def addTag(self, _tag: str) -> None:
        pass

    def removeTag(self, _tag: str) -> None:
        pass


def test_spark_table_source_uses_the_configured_limit() -> None:
    result = MaterializedResult()
    context = ArtifactContext(
        policy=ArtifactPolicy(
            table_max_rows=DEFAULT_TABLE_ARTIFACT_MAX_ROWS,
        ),
    )
    ref = context.create(
        SparkTableSource(
            spark=cast(Any, TaggedSpark()),
            dataframe=cast(Any, result),
        )
    )
    content = asyncio.run(context.materialize(ref))
    assert isinstance(content, TableContent)

    assert result.limit_value == DEFAULT_TABLE_ARTIFACT_MAX_ROWS + 1
    assert content.row_count == 1


def test_artifact_renderer_renders_a_published_artifact(
    make_table_artifact: Any,
) -> None:
    artifact_context = ArtifactContext()
    artifact = make_table_artifact([{"id": 1}])
    artifact_context.registry.add(artifact)
    reference = asyncio.run(artifact_context.publish(artifact.ref))
    container = TableContainer()
    renderer = ArtifactRenderer(artifact_context.registry)

    part = renderer.render_tool_result(reference, container)

    assert part is not None
    assert part.artifact_id == artifact.id
    assert isinstance(container.data, pd.DataFrame)
    assert container.download_button_calls == []


def test_artifact_renderer_rejects_mismatched_metadata(
    make_table_artifact: Any,
) -> None:
    artifact_context = ArtifactContext()
    artifact = make_table_artifact([{"id": 1}])
    artifact_context.registry.add(artifact)
    renderer = ArtifactRenderer(artifact_context.registry)

    with pytest.raises(TypeError, match="has kind 'table', not 'image'"):
        renderer.render_tool_result(
            ArtifactRef(
                artifact_id=artifact.id,
                artifact_kind="image",
            ),
            TableContainer(),
        )


def test_historical_arrow_snapshot_is_rendered_as_native_pandas_dataframe(
    make_table_artifact: Any,
) -> None:
    # Streamlit 1.60's frontend raises Array(-1) when it receives Arrow pandas
    # metadata with no column-index levels. Index-free snapshots have exactly
    # that metadata, so historical rendering must let Streamlit re-encode a
    # pandas DataFrame instead of passing the restored Arrow table directly.
    source = pa.table({f"column_{index}": [index] for index in range(244)})
    artifact = make_table_artifact(source)
    container = TableContainer()

    ArtifactRenderer(ArtifactRegistry()).render(container, artifact)

    assert isinstance(container.data, pd.DataFrame)
    assert container.data.shape == (1, 244)
    assert container.dataframe_kwargs == {
        "width": "stretch",
        "hide_index": True,
    }
    assert container.download_button_calls == []


def test_truncated_table_adds_csv_download(make_table_artifact: Any) -> None:
    artifact = make_table_artifact(
        [{"id": 1}, {"id": 2}],
        max_rows=1,
        artifact_id="truncated",
    )
    container = TableContainer()

    ArtifactRenderer(ArtifactRegistry()).render(
        container,
        artifact,
        key="download-truncated",
    )

    assert len(container.download_button_calls) == 1
    args, kwargs = container.download_button_calls[0]
    assert args == ("Download displayed rows as CSV", b"id\n1\n")
    assert kwargs == {
        "file_name": "secha-truncated.csv",
        "mime": "text/csv",
        "key": "download-truncated",
    }


class VisualizationContainer:
    def __init__(self) -> None:
        self.chart_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def vega_lite_chart(self, *args: Any, **kwargs: Any) -> None:
        self.chart_calls.append((args, kwargs))


def test_visualization_renderer_binds_named_data() -> None:
    spec = {
        "data": {"name": "source"},
        "datasets": {"source": [{"x": "A", "y": 1}]},
        "mark": "bar",
        "encoding": {
            "x": {"field": "x", "type": "nominal"},
            "y": {"field": "y", "type": "quantitative"},
        },
    }
    content = VisualizationContent(spec_json=json.dumps(spec).encode())
    ref = ArtifactRef(artifact_id="viz-1", artifact_kind="visualization")
    artifact = Artifact(ref=ref, content=content)
    container = VisualizationContainer()

    ArtifactRenderer(ArtifactRegistry()).render(
        container,
        artifact,
    )

    args, kwargs = container.chart_calls[0]
    assert args[0].to_dict(orient="records") == [{"x": "A", "y": 1}]
    assert "data" not in args[1]
    assert "datasets" not in args[1]
    assert kwargs == {"width": "stretch", "height": 500, "theme": None}
