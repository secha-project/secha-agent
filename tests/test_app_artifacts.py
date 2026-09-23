from typing import Any

from src.agent.outputs.artifacts import ArtifactCollector, ChartArtifact
from src.app.rendering import render_artifact, render_tool_result


class TableContainer:
    def __init__(self) -> None:
        self.data: Any = None
        self.download_button_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def columns(self, _spec: list[int], **_kwargs: Any) -> list["TableContainer"]:
        return [self, self]

    def dataframe(self, data: Any, **_kwargs: Any) -> None:
        self.data = data

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
    assert container.data.to_dict(orient="records") == [{"id": 1}]
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
    assert args[1] == b"id\n1\n"
    assert kwargs["mime"] == "text/csv"


class VisualizationContainer:
    def __init__(self) -> None:
        self.chart_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def columns(
        self, _spec: list[int], **_kwargs: Any
    ) -> list["VisualizationContainer"]:
        return [self, self]

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
    artifact = ChartArtifact(id="viz-1", spec=spec)
    container = VisualizationContainer()

    render_artifact(
        container,
        artifact,
    )

    args, _ = container.chart_calls[0]
    assert args[0].to_dict(orient="records") == [{"x": "A", "y": 1}]
    assert "data" not in args[1]
    assert "datasets" not in args[1]
    assert args[1]["mark"] == "bar"
    assert args[1]["encoding"] == spec["encoding"]
