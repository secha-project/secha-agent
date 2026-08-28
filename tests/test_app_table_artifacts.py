import asyncio
import csv
import io
from typing import Any, cast

import pyarrow as pa
from pydantic_ai import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ToolReturnPart,
)

from src.agent.outputs.artifacts import ArtifactRef, TableArtifact
from src.agent.outputs.tables import LiveTableOutput


class MaterializedResult:
    def __init__(self, table: pa.Table) -> None:
        self.table = table
        self.limit_value: int | None = None

    def limit(self, value: int) -> "MaterializedResult":
        self.limit_value = value
        return self

    def toPandas(self):
        table = (
            self.table
            if self.limit_value is None
            else self.table.slice(0, self.limit_value)
        )
        return table.to_pandas()


class TaggedSpark:
    def addTag(self, _tag: str) -> None:
        pass

    def removeTag(self, _tag: str) -> None:
        pass


def test_table_artifact_caps_and_round_trips_exact_arrow_values() -> None:
    source = pa.table(
        {
            "id": range(10_001),
            "nullable": [None, "value"] + ["other"] * 9_999,
            "nested": [[1, 2]] * 10_001,
        }
    )

    output = LiveTableOutput(
        spark=cast(Any, TaggedSpark()),
        dataframe=cast(Any, MaterializedResult(source)),
    )
    artifact = asyncio.run(output.to_artifact(10_000))
    assert isinstance(artifact, TableArtifact)
    restored = artifact.to_arrow_table()

    assert artifact.row_count == 10_000
    assert artifact.truncated is True
    assert restored.schema == source.schema
    assert restored["nullable"][0].as_py() is None
    assert restored["nested"][0].as_py() == [1, 2]
    assert restored["id"][-1].as_py() == 9_999


def test_csv_and_tool_metadata_are_json_safe(make_table_artifact) -> None:
    artifact = make_table_artifact(
        [{"id": 1, "value": "a,b"}, {"id": 2, "value": "two"}],
        max_rows=1,
    )
    rows = list(
        csv.DictReader(io.StringIO(artifact.to_csv().decode("utf-8")))
    )
    assert rows == [{"id": "1", "value": "a,b"}]

    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="display_table_output",
                    tool_call_id="call-1",
                    content="displayed",
                    metadata=ArtifactRef(artifact_id="artifact-17"),
                )
            ]
        )
    ]
    serialized = ModelMessagesTypeAdapter.dump_json(history)
    restored = ModelMessagesTypeAdapter.validate_json(serialized)
    part = restored[0].parts[0]
    assert isinstance(part, ToolReturnPart)
    assert part.metadata == {"artifact_id": "artifact-17"}
