from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pandas as pd
import pyarrow as pa
from pyarrow import ipc
from pydantic import BaseModel, ConfigDict
from pydantic_ai import ModelRetry
from pyspark.sql import DataFrame, SparkSession

from ..spark_runtime import (
    concise_spark_error,
    run_spark_with_timeout,
    spark_error_type,
)
from .artifacts import TableArtifact


class TableOutputRef(BaseModel):
    """Reference to a live table output available in the current session."""

    model_config = ConfigDict(frozen=True)

    output_id: str


class UnknownTableOutputError(LookupError):
    def __init__(self, ref: TableOutputRef) -> None:
        self.ref = ref
        super().__init__(f"Unknown table output {ref.output_id!r}.")


@dataclass(frozen=True)
class TablePreview:
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
class _PreviewSelection:
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
    ) -> _PreviewSelection:
        if start_row < 1:
            raise ValueError("start_row must be at least 1.")
        if end_row is not None and end_row < start_row:
            raise ValueError("end_row must be greater than or equal to start_row.")
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
            end_row=end_row if end_row is not None else start_row + 9,
            range_was_requested=end_row is not None,
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
class LiveTableOutput:
    """A live Spark table that tools can inspect, display, or visualize."""

    spark: SparkSession
    dataframe: DataFrame

    async def preview(
        self,
        start_row: int = 1,
        end_row: int | None = None,
        column_names: list[str] | None = None,
    ) -> TablePreview:
        available_columns = self.dataframe.schema.fieldNames()
        selection = _PreviewSelection.create(
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

        return await run_spark_with_timeout(self.spark, collect_preview)

    async def sample(self, max_rows: int) -> tuple[pd.DataFrame, bool]:
        frame = await run_spark_with_timeout(
            self.spark,
            self.dataframe.limit(max_rows + 1).toPandas,
        )
        return frame.iloc[:max_rows].reset_index(drop=True), len(frame.index) > max_rows

    async def to_artifact(self, max_rows: int | None) -> TableArtifact:
        dataframe = self.dataframe
        if max_rows is not None:
            dataframe = dataframe.limit(max_rows + 1)
        frame = await run_spark_with_timeout(self.spark, dataframe.toPandas)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        truncated = max_rows is not None and table.num_rows > max_rows
        visible = table if max_rows is None else table.slice(0, max_rows)
        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, visible.schema) as writer:
            writer.write_table(visible)
        return TableArtifact(
            arrow_bytes=sink.getvalue().to_pybytes(),
            truncated=truncated,
        )


class TableOutputStore:
    """Keep live table outputs addressable for the current conversation."""

    def __init__(self) -> None:
        self._outputs: dict[str, LiveTableOutput] = {}

    def add(self, output: LiveTableOutput) -> TableOutputRef:
        ref = TableOutputRef(output_id=uuid4().hex)
        self._outputs[ref.output_id] = output
        return ref

    def resolve(self, ref: TableOutputRef) -> LiveTableOutput:
        try:
            return self._outputs[ref.output_id]
        except KeyError as exc:
            raise UnknownTableOutputError(ref) from exc


class TableOutput(BaseModel):
    output_ref: TableOutputRef
    preview_markdown: str


async def create_table_output(
    store: TableOutputStore,
    spark: SparkSession,
    dataframe: DataFrame,
    *,
    error_prefix: str = "Table query failed while reading the preview",
) -> TableOutput:
    """Register and preview a Spark DataFrame as a shared live table output."""
    output = LiveTableOutput(spark=spark, dataframe=dataframe)
    try:
        preview = await output.preview()
    except Exception as exc:
        raise ModelRetry(
            f"{error_prefix} ({spark_error_type(exc)}): {concise_spark_error(exc)}."
        ) from exc

    ref = store.add(output)
    return TableOutput(output_ref=ref, preview_markdown=preview.to_markdown())
