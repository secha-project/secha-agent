from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal
from uuid import uuid4

import pandas as pd
import pyarrow as pa
from pyarrow import ipc
from pydantic import BaseModel, ConfigDict


class ArtifactRef(BaseModel):
    """Reference to a concrete artifact published by the current agent run."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str


@dataclass(frozen=True)
class TableArtifact:
    """Portable Arrow table displayed and persisted with a conversation."""

    MEDIA_TYPE: ClassVar[Literal["application/vnd.apache.arrow.stream"]] = (
        "application/vnd.apache.arrow.stream"
    )

    arrow_bytes: bytes
    truncated: bool
    id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def ref(self) -> ArtifactRef:
        return ArtifactRef(artifact_id=self.id)

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


@dataclass(frozen=True)
class ChartArtifact:
    """Portable, self-contained Vega-Lite chart."""

    MEDIA_TYPE: ClassVar[Literal["application/vnd.vegalite.v6+json"]] = (
        "application/vnd.vegalite.v6+json"
    )

    spec: dict[str, Any]
    id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def ref(self) -> ArtifactRef:
        return ArtifactRef(artifact_id=self.id)

    @property
    def spec_json(self) -> bytes:
        return (json.dumps(self.spec, indent=2, ensure_ascii=False) + "\n").encode()

    def to_spec(self) -> dict[str, Any]:
        return self.spec.copy()


type Artifact = TableArtifact | ChartArtifact


class UnknownArtifactError(LookupError):
    def __init__(self, ref: ArtifactRef) -> None:
        self.ref = ref
        super().__init__(f"Unknown artifact reference {ref.artifact_id!r}.")


class ArtifactCollector:
    """Collect concrete artifacts published during one agent run."""

    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}

    def publish(self, artifact: Artifact) -> ArtifactRef:
        existing = self._artifacts.get(artifact.id)
        if existing is not None and existing != artifact:
            raise ValueError(f"Artifact {artifact.id!r} is already published.")
        self._artifacts[artifact.id] = artifact
        return artifact.ref

    def resolve(self, ref: ArtifactRef) -> Artifact:
        try:
            return self._artifacts[ref.artifact_id]
        except KeyError as exc:
            raise UnknownArtifactError(ref) from exc

    @property
    def published(self) -> tuple[Artifact, ...]:
        return tuple(self._artifacts.values())
