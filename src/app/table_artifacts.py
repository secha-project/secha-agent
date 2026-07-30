from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pyarrow as pa
from pyarrow import ipc

MAX_DISPLAY_ROWS = 10_000


@dataclass(frozen=True)
class TableSnapshot:
    id: str
    arrow_bytes: bytes
    visible_row_count: int
    truncated: bool


def _as_arrow_table(value: Any) -> pa.Table:
    if isinstance(value, pa.Table):
        return value
    if isinstance(value, list) and all(isinstance(row, dict) for row in value):
        return pa.Table.from_pylist(value)

    # Spark's toPandas() returns a pandas DataFrame. Avoid importing pandas at
    # module import time because pyarrow can perform this conversion directly.
    if type(value).__module__.startswith("pandas"):
        return pa.Table.from_pandas(value, preserve_index=False)
    raise TypeError("Expected a PyArrow table, pandas DataFrame, or list of rows.")


def snapshot_table(
    value: Any,
    *,
    max_rows: int = MAX_DISPLAY_ROWS,
    artifact_id: str | None = None,
) -> TableSnapshot:
    """Cap a materialized result and serialize the exact display snapshot."""
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1.")

    table = _as_arrow_table(value)
    truncated = table.num_rows > max_rows
    visible = table.slice(0, max_rows)
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, visible.schema) as writer:
        writer.write_table(visible)
    return TableSnapshot(
        id=artifact_id or uuid4().hex,
        arrow_bytes=sink.getvalue().to_pybytes(),
        visible_row_count=visible.num_rows,
        truncated=truncated,
    )


def arrow_bytes_to_table(value: bytes) -> pa.Table:
    with ipc.open_stream(pa.BufferReader(value)) as reader:
        return reader.read_all()


def snapshot_to_csv(snapshot: TableSnapshot) -> bytes:
    table = arrow_bytes_to_table(snapshot.arrow_bytes)
    output = io.StringIO()
    table.to_pandas().to_csv(output, index=False)
    return output.getvalue().encode("utf-8")
