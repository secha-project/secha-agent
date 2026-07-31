import csv
import io

import pyarrow as pa
from pydantic_ai import ModelMessagesTypeAdapter, ModelRequest, ToolReturnPart

from src.app.table_artifacts import (
    arrow_bytes_to_table,
    snapshot_table,
    snapshot_to_csv,
)


def test_snapshot_caps_and_round_trips_exact_arrow_values() -> None:
    source = pa.table(
        {
            "id": range(10_001),
            "nullable": [None, "value"] + ["other"] * 9_999,
            "nested": [[1, 2]] * 10_001,
        }
    )

    snapshot = snapshot_table(source)
    restored = arrow_bytes_to_table(snapshot.arrow_bytes)

    assert snapshot.visible_row_count == 10_000
    assert snapshot.truncated is True
    assert restored.schema == source.schema
    assert restored["nullable"][0].as_py() is None
    assert restored["nested"][0].as_py() == [1, 2]
    assert restored["id"][-1].as_py() == 9_999


def test_csv_and_tool_metadata_are_json_safe() -> None:
    snapshot = snapshot_table(
        [{"id": 1, "value": "a,b"}, {"id": 2, "value": "two"}],
        max_rows=1,
    )
    rows = list(
        csv.DictReader(io.StringIO(snapshot_to_csv(snapshot).decode("utf-8")))
    )
    assert rows == [{"id": "1", "value": "a,b"}]

    history = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="display_query_result",
                    tool_call_id="call-1",
                    content="displayed",
                    metadata={
                        "ui_kind": "dataframe",
                        "dataframe_ref": "df-17",
                    },
                )
            ]
        )
    ]
    serialized = ModelMessagesTypeAdapter.dump_json(history)
    restored = ModelMessagesTypeAdapter.validate_json(serialized)
    part = restored[0].parts[0]
    assert isinstance(part, ToolReturnPart)
    assert part.metadata == {
        "ui_kind": "dataframe",
        "dataframe_ref": "df-17",
    }
