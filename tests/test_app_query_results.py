import asyncio
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

from src.app.query_results import (
    ResultDisplayHandler,
    render_snapshot,
)
from src.app.table_artifacts import MAX_DISPLAY_ROWS, snapshot_table


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

    def toPandas(self) -> list[dict[str, int]]:
        return [{"id": 1}]


class TaggedSpark:
    def addTag(self, _tag: str) -> None:
        pass

    def removeTag(self, _tag: str) -> None:
        pass


class ResultRegistryStub:
    def __init__(self, result: MaterializedResult) -> None:
        self.result = result

    def resolve(self, ref: str) -> MaterializedResult:
        assert ref == "df-1"
        return self.result


def test_result_display_handler_uses_the_shared_limit_and_snapshot() -> None:
    result = MaterializedResult()
    registry = ResultRegistryStub(result)
    handler = ResultDisplayHandler(TaggedSpark(), registry)  # type: ignore[arg-type]
    container = TableContainer()

    part = asyncio.run(
        handler.render_tool_result(
            {"ui_kind": "dataframe", "dataframe_ref": "df-1"},
            container,
        )
    )

    assert part is not None
    assert result.limit_value == MAX_DISPLAY_ROWS + 1
    assert part.artifact_id == handler.snapshots[0].id
    snapshot = handler.snapshots[0]
    assert snapshot.visible_row_count == 1
    assert isinstance(container.data, pd.DataFrame)
    assert container.download_button_calls == []


def test_result_display_handler_rejects_a_non_string_reference() -> None:
    handler = ResultDisplayHandler(
        TaggedSpark(),  # type: ignore[arg-type]
        ResultRegistryStub(MaterializedResult()),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError):
        asyncio.run(
            handler.render_tool_result(
                {"ui_kind": "dataframe", "dataframe_ref": 1},
                TableContainer(),
            )
        )


def test_historical_arrow_snapshot_is_rendered_as_native_pandas_dataframe() -> None:
    # Streamlit 1.60's frontend raises Array(-1) when it receives Arrow pandas
    # metadata with no column-index levels. Index-free snapshots have exactly
    # that metadata, so historical rendering must let Streamlit re-encode a
    # pandas DataFrame instead of passing the restored Arrow table directly.
    source = pa.table({f"column_{index}": [index] for index in range(244)})
    snapshot = snapshot_table(source)
    container = TableContainer()

    render_snapshot(container, snapshot)

    assert isinstance(container.data, pd.DataFrame)
    assert container.data.shape == (1, 244)
    assert container.dataframe_kwargs == {
        "width": "stretch",
        "hide_index": True,
    }
    assert container.download_button_calls == []


def test_truncated_table_adds_csv_download() -> None:
    snapshot = snapshot_table(
        [{"id": 1}, {"id": 2}],
        max_rows=1,
        artifact_id="truncated",
    )
    container = TableContainer()

    render_snapshot(container, snapshot, key="download-truncated")

    assert len(container.download_button_calls) == 1
    args, kwargs = container.download_button_calls[0]
    assert args == ("Download displayed rows as CSV", b"id\n1\n")
    assert kwargs == {
        "file_name": "secha-truncated.csv",
        "mime": "text/csv",
        "key": "download-truncated",
    }
