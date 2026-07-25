from contextvars import Token
from pathlib import Path

from pydantic_evals.lifecycle import CaseLifecycle

from src.agent.agent import reset_evaluation_csv_path, set_evaluation_csv_path
from src.validation.eval_models import BenchmarkCase, BenchmarkOutput, CaseMetadata


class EvaluationLifecycle(CaseLifecycle[str, BenchmarkOutput, CaseMetadata]):
    """Configure the CSV output path for one evaluation case."""

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
        self.csv_path_token: Token[Path | None] | None = None

    async def setup(self) -> None:
        task_name = self.case.name
        if task_name is None:
            raise ValueError("Evaluation cases must have names to store CSV artifacts.")

        artifact_name = (
            task_name
            if self.repetition_index is None
            else f"{task_name}_{self.repetition_index}"
        )
        filename = f"{artifact_name}.csv"
        if Path(filename).name != filename:
            raise ValueError(
                f"Evaluation case name {task_name!r} cannot be used "
                "as an artifact name."
            )
        self.csv_path_token = set_evaluation_csv_path(self.artifacts_dir / filename)

    async def teardown(self, result: object) -> None:
        if self.csv_path_token is not None:
            reset_evaluation_csv_path(self.csv_path_token)
            self.csv_path_token = None


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
