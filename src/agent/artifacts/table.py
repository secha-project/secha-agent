from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import pandas as pd
import pyarrow as pa
from pyarrow import ipc
from pyspark.sql import DataFrame, SparkSession

from ..spark_runtime import run_spark_with_timeout
from .base import RunArtifacts


@dataclass(frozen=True)
class TablePreview:
    """A selected view of a table, independent of its live representation."""

    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    total_rows: int
    total_columns: int
    start_row: int
    requested_row_count: int
    range_was_requested: bool
    columns_were_requested: bool

    def to_markdown(self) -> str:
        sections = []
        for row_number, row in enumerate(self.rows, start=self.start_row):
            fields = "\n".join(
                f"{column}: {value}" for column, value in zip(self.columns, row)
            )
            sections.append(f"## Row {row_number}\n\n{fields}")
        rendered_rows = "\n\n".join(sections)

        overflow_messages = []
        if self.total_rows > self.requested_row_count:
            row_selection = "requested" if self.range_was_requested else "default"
            row_label = "row" if len(self.rows) == 1 else "rows"
            overflow_messages.append(
                f"showing only the {row_selection} {len(self.rows)} {row_label} "
                f"out of the {self.total_rows} available"
            )
        if len(self.columns) < self.total_columns:
            column_selection = "requested" if self.columns_were_requested else "default"
            column_label = "column" if len(self.columns) == 1 else "columns"
            overflow_messages.append(
                f"showing only the {column_selection} {len(self.columns)} "
                f"{column_label} out of the {self.total_columns} available"
            )

        if not overflow_messages:
            return rendered_rows

        footer = f"[{', '.join(overflow_messages)}]"
        return f"{rendered_rows}\n\n{footer}" if rendered_rows else footer


@dataclass(frozen=True)
class _TablePreviewSelection:
    columns: tuple[str, ...]
    start_row: int
    end_row: int
    range_was_requested: bool
    columns_were_requested: bool

    @classmethod
    def create(
        cls,
        available_columns: list[str],
        *,
        start_row: int,
        end_row: int | None,
        column_names: list[str] | None,
    ) -> _TablePreviewSelection:
        range_was_requested = end_row is not None
        if start_row < 1:
            raise ValueError("start_row must be at least 1.")
        if end_row is not None and end_row < start_row:
            raise ValueError("end_row must be greater than or equal to start_row.")
        resolved_end_row = end_row if end_row is not None else start_row + 9
        selected_columns = column_names or available_columns[:10]
        unknown_columns = [
            column for column in selected_columns if column not in available_columns
        ]
        if unknown_columns:
            raise ValueError(
                f"Unknown result columns: {', '.join(unknown_columns)}. "
                f"Available columns: {', '.join(available_columns)}."
            )
        return cls(
            columns=tuple(selected_columns),
            start_row=start_row,
            end_row=resolved_end_row,
            range_was_requested=range_was_requested,
            columns_were_requested=column_names is not None,
        )

    @property
    def requested_row_count(self) -> int:
        return self.end_row - self.start_row + 1

    def build(
        self,
        rows: tuple[tuple[object, ...], ...],
        *,
        total_rows: int,
        total_columns: int,
    ) -> TablePreview:
        return TablePreview(
            columns=self.columns,
            rows=rows,
            total_rows=total_rows,
            total_columns=total_columns,
            start_row=self.start_row,
            requested_row_count=self.requested_row_count,
            range_was_requested=self.range_was_requested,
            columns_were_requested=self.columns_were_requested,
        )


@dataclass(frozen=True)
class TableContent:
    """Portable, materialized content of a table artifact."""

    MEDIA_TYPE: ClassVar[Literal["application/vnd.apache.arrow.stream"]] = (
        "application/vnd.apache.arrow.stream"
    )

    arrow_bytes: bytes
    truncated: bool
    kind: Literal["table"] = field(default="table", init=False)
    media_type: Literal["application/vnd.apache.arrow.stream"] = field(
        default=MEDIA_TYPE,
        init=False,
    )

    @property
    def row_count(self) -> int:
        return self.to_arrow_table().num_rows

    def to_arrow_table(self) -> pa.Table:
        with ipc.open_stream(pa.BufferReader(self.arrow_bytes)) as reader:
            return reader.read_all()

    def to_pandas(self) -> pd.DataFrame:
        return self.to_arrow_table().to_pandas()

    def to_csv(self) -> bytes:
        output = io.StringIO()
        self.to_pandas().to_csv(output, index=False)
        return output.getvalue().encode("utf-8")

    def preview(
        self,
        start_row: int = 1,
        end_row: int | None = None,
        column_names: list[str] | None = None,
    ) -> TablePreview:
        table = self.to_arrow_table()
        selection = _TablePreviewSelection.create(
            table.column_names,
            start_row=start_row,
            end_row=end_row,
            column_names=column_names,
        )
        selected = table.select(selection.columns)
        visible = selected.slice(
            selection.start_row - 1,
            selection.requested_row_count,
        )
        rows = tuple(
            tuple(row[column] for column in selection.columns)
            for row in visible.to_pylist()
        )
        return selection.build(
            rows,
            total_rows=table.num_rows,
            total_columns=table.num_columns,
        )


@dataclass(frozen=True)
class SparkTableSource:
    """A live Spark result that can be inspected or materialized as a table."""

    spark: SparkSession
    dataframe: DataFrame
    kind: Literal["table"] = field(default="table", init=False)

    async def preview(
        self,
        start_row: int = 1,
        end_row: int | None = None,
        column_names: list[str] | None = None,
    ) -> TablePreview:
        available_columns = self.dataframe.schema.fieldNames()
        selection = _TablePreviewSelection.create(
            available_columns,
            start_row=start_row,
            end_row=end_row,
            column_names=column_names,
        )

        def collect_preview() -> TablePreview:
            total_rows = self.dataframe.count()
            collected = (
                self.dataframe.select(*selection.columns)
                .limit(selection.end_row)
                .collect()
            )
            rows = tuple(
                tuple(row)
                for row in collected[selection.start_row - 1 : selection.end_row]
            )
            return selection.build(
                rows,
                total_rows=total_rows,
                total_columns=len(available_columns),
            )

        return await run_spark_with_timeout(
            self.spark,
            collect_preview,
        )

    async def materialize(self, artifacts: RunArtifacts) -> TableContent:
        max_rows = artifacts.policy.table_max_rows
        dataframe = self.dataframe
        if max_rows is not None:
            dataframe = dataframe.limit(max_rows + 1)
        materialized = await run_spark_with_timeout(
            self.spark,
            dataframe.toPandas,
        )
        table = pa.Table.from_pandas(materialized, preserve_index=False)
        truncated = max_rows is not None and table.num_rows > max_rows
        visible = table if max_rows is None else table.slice(0, max_rows)
        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, visible.schema) as writer:
            writer.write_table(visible)
        return TableContent(
            arrow_bytes=sink.getvalue().to_pybytes(),
            truncated=truncated,
        )
