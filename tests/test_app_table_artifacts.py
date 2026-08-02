import asyncio
import csv
import io

import pyarrow as pa
from pydantic_ai import ModelMessagesTypeAdapter, ModelRequest, ToolReturnPart

from src.agent.artifacts.base import (
    ArtifactContext,
    ArtifactPolicy,
    ArtifactRef,
)
from src.agent.artifacts.table import SparkTableSource


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

    context = ArtifactContext(
        policy=ArtifactPolicy(table_max_rows=10_000),
    )
    ref = context.create(
        SparkTableSource(
            spark=TaggedSpark(),  # type: ignore[arg-type]
            dataframe=MaterializedResult(source),  # type: ignore[arg-type]
        )
    )
    content = asyncio.run(context.materialize(ref))
    restored = content.to_arrow_table()

    assert content.row_count == 10_000
    assert content.truncated is True
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
        csv.DictReader(
            io.StringIO(
                artifact.require_content().to_csv().decode("utf-8")
            )
        )
    )
    assert rows == [{"id": "1", "value": "a,b"}]

    history = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="display_query_result",
                    tool_call_id="call-1",
                    content="displayed",
                    metadata=ArtifactRef(
                        artifact_id="artifact-17",
                        artifact_kind="table",
                    ),
                )
            ]
        )
    ]
    serialized = ModelMessagesTypeAdapter.dump_json(history)
    restored = ModelMessagesTypeAdapter.validate_json(serialized)
    part = restored[0].parts[0]
    assert isinstance(part, ToolReturnPart)
    assert part.metadata == {
        "artifact_id": "artifact-17",
        "artifact_kind": "table",
    }


def test_materialized_table_content_produces_a_preview(make_table_artifact) -> None:
    artifact = make_table_artifact(
        [
            {"id": 1, "value": "first"},
            {"id": 2, "value": "second"},
        ]
    )

    preview = artifact.require_content().preview(
        start_row=2,
        end_row=2,
        column_names=["value"],
    )

    assert preview.rows == (("second",),)
    assert preview.to_markdown() == (
        "## Row 2\n\nvalue: second\n\n"
        "[showing only the requested 1 row out of the 2 available, "
        "showing only the requested 1 column out of the 2 available]"
    )
