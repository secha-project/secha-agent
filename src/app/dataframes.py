from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

import pyarrow as pa
import pyarrow.ipc as ipc
from pydantic_ai import ModelRetry
from pyspark.sql import DataFrame, SparkSession

MAX_DISPLAY_ROWS = 10_000


class DataFrameQueryRepository(Protocol):
    def save_dataframe_query(self, conversation_id: str, sql: str) -> int: ...

    def load_dataframe_query(
        self,
        conversation_id: str,
        query_id: int,
    ) -> str | None: ...


class DataFrameReferences:
    """Conversation-scoped live Spark plans backed by durable SQL recipes."""

    _REFERENCE_PATTERN = re.compile(r"df-([1-9][0-9]*)")

    def __init__(
        self,
        spark: SparkSession,
        repository: DataFrameQueryRepository,
        conversation_id: str,
    ) -> None:
        self.spark = spark
        self.repository = repository
        self.conversation_id = conversation_id
        self.live: dict[str, DataFrame] = {}

    def create(self, sql: str, dataframe: DataFrame) -> str:
        query_id = self.repository.save_dataframe_query(self.conversation_id, sql)
        ref = f"df-{query_id}"
        self.live[ref] = dataframe
        return ref

    def resolve(self, ref: str) -> DataFrame:
        cached = self.live.get(ref)
        if cached is not None:
            return cached

        match = self._REFERENCE_PATTERN.fullmatch(ref)
        if match is None:
            raise ModelRetry(
                f"Invalid dataframe reference {ref!r}. Use a reference such as 'df-17'."
            )

        sql = self.repository.load_dataframe_query(
            self.conversation_id,
            int(match.group(1)),
        )
        if sql is None:
            raise ModelRetry(
                f"Dataframe reference {ref!r} does not belong to this conversation."
            )

        try:
            dataframe = self.spark.sql(sql)
        except Exception as exc:
            raise ModelRetry(
                f"Could not recreate dataframe reference {ref!r}: {exc}"
            ) from exc
        self.live[ref] = dataframe
        return dataframe

    def clear(self) -> None:
        self.live.clear()


@dataclass(frozen=True)
class DataframeArtifact:
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
) -> DataframeArtifact:
    """Cap a materialized result and serialize the exact display snapshot."""
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1.")

    table = _as_arrow_table(value)
    truncated = table.num_rows > max_rows
    visible = table.slice(0, max_rows)
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, visible.schema) as writer:
        writer.write_table(visible)
    return DataframeArtifact(
        id=artifact_id or uuid4().hex,
        arrow_bytes=sink.getvalue().to_pybytes(),
        visible_row_count=visible.num_rows,
        truncated=truncated,
    )


def arrow_bytes_to_table(value: bytes) -> pa.Table:
    with ipc.open_stream(pa.BufferReader(value)) as reader:
        return reader.read_all()


def artifact_to_csv(artifact: DataframeArtifact) -> bytes:
    table = arrow_bytes_to_table(artifact.arrow_bytes)
    output = io.StringIO()
    table.to_pandas().to_csv(output, index=False)
    return output.getvalue().encode("utf-8")
