from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from .artifacts import ChartArtifact


class ChartOutputRef(BaseModel):
    """Reference to a chart preview available in the current agent run."""

    model_config = ConfigDict(frozen=True)

    output_id: str


class UnknownChartOutputError(LookupError):
    def __init__(self, ref: ChartOutputRef) -> None:
        self.ref = ref
        super().__init__(f"Unknown chart output {ref.output_id!r}.")


class ChartOutputStore:
    """Keep chart drafts separate from artifacts published to the user."""

    def __init__(self) -> None:
        self._outputs: dict[str, ChartArtifact] = {}

    def add(self, artifact: ChartArtifact) -> ChartOutputRef:
        ref = ChartOutputRef(output_id=uuid4().hex)
        self._outputs[ref.output_id] = artifact
        return ref

    def resolve(self, ref: ChartOutputRef) -> ChartArtifact:
        try:
            return self._outputs[ref.output_id]
        except KeyError as exc:
            raise UnknownChartOutputError(ref) from exc


class ChartOutput(BaseModel):
    output_ref: ChartOutputRef
    truncated: bool
