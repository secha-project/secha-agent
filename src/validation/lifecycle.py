from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path

from pydantic_evals import set_eval_attribute
from pydantic_evals.lifecycle import CaseLifecycle

from src.agent.artifacts.base import Artifact
from src.agent.artifacts.table import TableContent
from src.validation.eval_models import BenchmarkCase, BenchmarkOutput, CaseMetadata


@dataclass(frozen=True)
class EvaluationArtifactScope:
    directory: Path
    basename: str


_evaluation_artifact_scope = ContextVar[EvaluationArtifactScope | None](
    "evaluation_artifact_scope",
    default=None,
)


def write_evaluation_artifacts(
    artifacts: tuple[Artifact, ...],
) -> tuple[Path, ...]:
    """Write one successful evaluation run's in-memory artifacts to disk."""
    if not artifacts:
        return ()

    scope = _evaluation_artifact_scope.get()
    if scope is None:
        raise RuntimeError("No artifact scope was configured for this evaluation case.")

    paths: list[Path] = []
    table_paths: list[Path] = []
    for index, artifact in enumerate(artifacts, start=1):
        suffix = "" if index == 1 else f"_{index}"
        content = artifact.require_content()
        if isinstance(content, TableContent):
            path = scope.directory / f"{scope.basename}{suffix}.csv"
            path.write_bytes(content.to_csv())
            table_paths.append(path)
        else:
            raise TypeError(
                f"Evaluation output does not support {artifact.kind!r} artifacts yet."
            )
        paths.append(path)

    set_eval_attribute("artifact_paths", [str(path) for path in paths])
    if table_paths:
        # Keep the existing evaluator contract while the richer manifest allows
        # additional artifacts to be added without another lifecycle redesign.
        set_eval_attribute("csv_path", str(table_paths[0]))
    return tuple(paths)


class EvaluationLifecycle(CaseLifecycle[str, BenchmarkOutput, CaseMetadata]):
    """Configure the artifact scope for one concurrent evaluation case."""

    def __init__(
        self,
        case: BenchmarkCase,
        *,
        artifacts_dir: Path,
        repetition_index: int | None = None,
    ) -> None:
        super().__init__(case)
        self.artifacts_dir = artifacts_dir
        self.repetition_index = repetition_index
        self.artifact_scope_token: Token[EvaluationArtifactScope | None] | None = None

    async def setup(self) -> None:
        task_name = self.case.name
        if task_name is None:
            raise ValueError("Evaluation cases must have names to store artifacts.")

        artifact_name = (
            task_name
            if self.repetition_index is None
            else f"{task_name}_{self.repetition_index}"
        )
        if Path(artifact_name).name != artifact_name:
            raise ValueError(
                f"Evaluation case name {task_name!r} cannot be used "
                "as an artifact name."
            )
        self.artifact_scope_token = _evaluation_artifact_scope.set(
            EvaluationArtifactScope(
                directory=self.artifacts_dir,
                basename=artifact_name,
            )
        )

    async def teardown(self, result: object) -> None:
        if self.artifact_scope_token is not None:
            _evaluation_artifact_scope.reset(self.artifact_scope_token)
            self.artifact_scope_token = None


class LifecycleFactory:
    """Create case lifecycles and number repeated case artifacts."""

    def __init__(self, artifacts_dir: Path, repetitions: int) -> None:
        self.artifacts_dir = artifacts_dir
        self.number_repetitions = repetitions > 1
        self.repetition_counts: dict[str | None, int] = {}

    def __call__(self, case: BenchmarkCase) -> EvaluationLifecycle:
        repetition_index = None
        if self.number_repetitions:
            repetition_index = self.repetition_counts.get(case.name, 0) + 1
            self.repetition_counts[case.name] = repetition_index

        return EvaluationLifecycle(
            case,
            artifacts_dir=self.artifacts_dir,
            repetition_index=repetition_index,
        )
