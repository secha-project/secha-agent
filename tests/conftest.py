from collections.abc import Callable
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pytest
from pyarrow import ipc

from src.agent.artifacts.base import (
    DEFAULT_TABLE_ARTIFACT_MAX_ROWS,
    Artifact,
    ArtifactRef,
)
from src.agent.artifacts.table import TableContent


@pytest.fixture
def make_table_artifact() -> Callable[..., Artifact]:
    """Build table artifacts directly for tests outside the Spark source path."""

    def make(
        value: Any,
        *,
        max_rows: int | None = DEFAULT_TABLE_ARTIFACT_MAX_ROWS,
        artifact_id: str | None = None,
    ) -> Artifact:
        table = value if isinstance(value, pa.Table) else pa.Table.from_pylist(value)
        truncated = max_rows is not None and table.num_rows > max_rows
        visible = table if max_rows is None else table.slice(0, max_rows)
        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, visible.schema) as writer:
            writer.write_table(visible)
        return Artifact(
            ref=ArtifactRef(
                artifact_id=artifact_id or uuid4().hex,
                artifact_kind="table",
            ),
            content=TableContent(
                arrow_bytes=sink.getvalue().to_pybytes(),
                truncated=truncated,
            ),
        )

    return make
