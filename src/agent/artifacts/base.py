from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

DEFAULT_TABLE_ARTIFACT_MAX_ROWS = 10_000

type ArtifactKind = Literal["table", "image", "visualization"]


class ArtifactRef(BaseModel):
    """Stable identity for an artifact in every representation."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str
    artifact_kind: ArtifactKind


class ArtifactContent(Protocol):
    """Portable content produced by materializing an artifact source."""

    @property
    def kind(self) -> ArtifactKind: ...


@dataclass(frozen=True)
class ArtifactPolicy:
    """Execution-specific choices used while materializing artifacts."""

    table_max_rows: int | None = DEFAULT_TABLE_ARTIFACT_MAX_ROWS

    def __post_init__(self) -> None:
        if self.table_max_rows is not None and self.table_max_rows < 1:
            raise ValueError("table_max_rows must be at least 1 or None.")


class ArtifactSource(Protocol):
    """An ephemeral recipe for producing an artifact's portable content."""

    @property
    def kind(self) -> ArtifactKind: ...

    async def materialize(
        self,
        context: ArtifactContext,
    ) -> ArtifactContent: ...


@dataclass
class Artifact:
    """One logical artifact with optional live and materialized representations."""

    ref: ArtifactRef
    source: ArtifactSource | None = field(default=None, compare=False, repr=False)
    content: ArtifactContent | None = None

    def __post_init__(self) -> None:
        if self.source is None and self.content is None:
            raise ValueError("An artifact must have a source or materialized content.")
        if self.source is not None and self.source.kind != self.ref.artifact_kind:
            raise ValueError(
                f"Artifact {self.ref.artifact_id!r} has kind "
                f"{self.ref.artifact_kind!r}, but its source has kind "
                f"{self.source.kind!r}."
            )
        if self.content is not None and self.content.kind != self.ref.artifact_kind:
            raise ValueError(
                f"Artifact {self.ref.artifact_id!r} has kind "
                f"{self.ref.artifact_kind!r}, but its content has kind "
                f"{self.content.kind!r}."
            )

    @property
    def id(self) -> str:
        return self.ref.artifact_id

    @property
    def kind(self) -> ArtifactKind:
        return self.ref.artifact_kind

    def require_content(self) -> ArtifactContent:
        if self.content is None:
            raise ArtifactContentUnavailableError(self.ref)
        return self.content


class UnknownArtifactError(LookupError):
    def __init__(self, ref: ArtifactRef) -> None:
        self.ref = ref
        super().__init__(f"Unknown artifact reference {ref.artifact_id!r}.")


class ArtifactSourceUnavailableError(LookupError):
    def __init__(self, ref: ArtifactRef) -> None:
        self.ref = ref
        super().__init__(
            f"The live source for artifact {ref.artifact_id!r} is unavailable."
        )


class ArtifactContentUnavailableError(LookupError):
    def __init__(self, ref: ArtifactRef) -> None:
        self.ref = ref
        super().__init__(f"Artifact {ref.artifact_id!r} has not been materialized.")


class UnexpectedArtifactSourceTypeError(TypeError):
    def __init__(
        self,
        ref: ArtifactRef,
        expected: type[object],
        actual: object,
    ) -> None:
        self.ref = ref
        super().__init__(
            f"Artifact {ref.artifact_id!r} has source "
            f"{type(actual).__name__}, not {expected.__name__}."
        )


class ArtifactDependencyCycleError(RuntimeError):
    def __init__(self, ref: ArtifactRef) -> None:
        self.ref = ref
        super().__init__(
            f"Materializing artifact {ref.artifact_id!r} would create a "
            "dependency cycle."
        )


ArtifactSourceT = TypeVar("ArtifactSourceT", bound=ArtifactSource)


class ArtifactRegistry:
    """Keep the live representations of logical artifacts addressable."""

    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}

    def create(self, source: ArtifactSource) -> ArtifactRef:
        ref = ArtifactRef(
            artifact_id=uuid4().hex,
            artifact_kind=source.kind,
        )
        self.add(Artifact(ref=ref, source=source))
        return ref

    def create_content(self, content: ArtifactContent) -> ArtifactRef:
        ref = ArtifactRef(
            artifact_id=uuid4().hex,
            artifact_kind=content.kind,
        )
        return self.add(Artifact(ref=ref, content=content))

    def add(self, artifact: Artifact) -> ArtifactRef:
        if artifact.id in self._artifacts:
            raise ValueError(f"Artifact {artifact.id!r} is already registered.")
        self._artifacts[artifact.id] = artifact
        return artifact.ref

    def resolve(self, ref: ArtifactRef) -> Artifact:
        try:
            artifact = self._artifacts[ref.artifact_id]
        except KeyError as exc:
            raise UnknownArtifactError(ref) from exc
        if artifact.kind != ref.artifact_kind:
            raise TypeError(
                f"Artifact {ref.artifact_id!r} has kind {artifact.kind!r}, "
                f"not {ref.artifact_kind!r}."
            )
        return artifact

    def resolve_source(
        self,
        ref: ArtifactRef,
        expected_type: type[ArtifactSourceT],
    ) -> ArtifactSourceT:
        source = self.resolve(ref).source
        if source is None:
            raise ArtifactSourceUnavailableError(ref)
        if not isinstance(source, expected_type):
            raise UnexpectedArtifactSourceTypeError(ref, expected_type, source)
        return cast(ArtifactSourceT, source)

    def set_content(
        self,
        ref: ArtifactRef,
        content: ArtifactContent,
    ) -> Artifact:
        artifact = self.resolve(ref)
        if content.kind != artifact.kind:
            raise TypeError(
                f"Artifact {artifact.id!r} has kind {artifact.kind!r}, "
                f"but materialization produced {content.kind!r}."
            )
        if artifact.content is not None and artifact.content != content:
            raise ValueError(
                f"Artifact {artifact.id!r} already has different materialized content."
            )
        artifact.content = content
        return artifact


class ArtifactContext:
    """Resolve and publish artifacts during one agent execution."""

    def __init__(
        self,
        registry: ArtifactRegistry | None = None,
        policy: ArtifactPolicy | None = None,
    ) -> None:
        self.registry = registry if registry is not None else ArtifactRegistry()
        self.policy = policy if policy is not None else ArtifactPolicy()
        self._published: dict[str, ArtifactRef] = {}
        self._materializing: set[str] = set()

    def create(self, source: ArtifactSource) -> ArtifactRef:
        return self.registry.create(source)

    def create_content(self, content: ArtifactContent) -> ArtifactRef:
        return self.registry.create_content(content)

    def resolve_source(
        self,
        ref: ArtifactRef,
        expected_type: type[ArtifactSourceT],
    ) -> ArtifactSourceT:
        return self.registry.resolve_source(ref, expected_type)

    async def materialize(self, ref: ArtifactRef) -> ArtifactContent:
        artifact = self.registry.resolve(ref)
        if artifact.content is not None:
            return artifact.content
        if artifact.source is None:
            raise ArtifactSourceUnavailableError(ref)
        if artifact.id in self._materializing:
            raise ArtifactDependencyCycleError(ref)

        self._materializing.add(artifact.id)
        try:
            content = await artifact.source.materialize(self)
            self.registry.set_content(ref, content)
            return content
        finally:
            self._materializing.remove(artifact.id)

    async def publish(self, ref: ArtifactRef) -> ArtifactRef:
        """Materialize an artifact and include it in this execution's output."""
        await self.materialize(ref)
        self._published.setdefault(ref.artifact_id, ref)
        return ref

    @property
    def published_artifacts(self) -> tuple[Artifact, ...]:
        return tuple(self.registry.resolve(ref) for ref in self._published.values())
