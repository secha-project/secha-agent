from typing import Any

import pandas as pd

from src.agent.outputs.artifacts import ArtifactCollector, ChartArtifact
from src.app.rendering import render_artifact, render_tool_result


class TableContainer:
    def __init__(self) -> None:
        self.data: Any = None
        self.dataframe_kwargs: dict[str, Any] = {}
        self.download_button_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.columns_spec: list[int] | None = None

    def columns(self, spec: list[int], **_kwargs: Any) -> list["TableContainer"]:
        self.columns_spec = spec
        return [self, self]

    def dataframe(self, data: Any, **kwargs: Any) -> None:
        self.data = data
        self.dataframe_kwargs = kwargs

    def caption(self, _value: str) -> None:
        pass

    def download_button(self, *args: Any, **kwargs: Any) -> None:
        self.download_button_calls.append((args, kwargs))


def test_artifact_renderer_renders_a_published_artifact(
    make_table_artifact: Any,
) -> None:
    artifacts = ArtifactCollector()
    artifact = make_table_artifact([{"id": 1}])
    reference = artifacts.publish(artifact)
    container = TableContainer()
    part = render_tool_result(reference, artifacts, container)

    assert part is not None
    assert part.artifact_id == artifact.id
    assert isinstance(container.data, pd.DataFrame)
    assert container.columns_spec == [9, 1]
    assert container.download_button_calls == []


def test_truncated_table_adds_csv_download(make_table_artifact: Any) -> None:
    artifact = make_table_artifact(
        [{"id": 1}, {"id": 2}],
        max_rows=1,
        artifact_id="truncated",
    )
    container = TableContainer()

    render_artifact(
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
        self.columns_spec: list[int] | None = None

    def columns(
        self, spec: list[int], **_kwargs: Any
    ) -> list["VisualizationContainer"]:
        self.columns_spec = spec
        return [self, self]

    def vega_lite_chart(self, *args: Any, **kwargs: Any) -> None:
        self.chart_calls.append((args, kwargs))

def test_visualization_renderer_binds_named_data() -> None:
    spec = {
        "data": {"name": "source"},
        "datasets": {"source": [{"x": "A", "y": 1}]},
        "mark": "bar",
        "title": "A chart title long enough to require native Vega wrapping",
        "encoding": {
            "x": {"field": "x", "type": "nominal"},
            "y": {"field": "y", "type": "quantitative"},
        },
    }
    artifact = ChartArtifact(id="viz-1", spec=spec)
    container = VisualizationContainer()

    render_artifact(
        container,
        artifact,
    )

    args, kwargs = container.chart_calls[0]
    assert args[0].to_dict(orient="records") == [{"x": "A", "y": 1}]
    assert "data" not in args[1]
    assert "datasets" not in args[1]
    assert args[1]["title"] == {
        "text": ["A chart title long enough to require native", "Vega wrapping"],
        "anchor": "middle",
    }
    assert container.columns_spec == [9, 1]
    assert kwargs == {"width": "stretch", "height": 500, "theme": None}
