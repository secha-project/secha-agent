"""Report-level accuracy and pass@k metrics for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from pydantic_evals.evaluators import ReportEvaluator, ReportEvaluatorContext
from pydantic_evals.reporting import EvaluationReport
from pydantic_evals.reporting.analyses import ScalarResult, TableResult

from .eval_models import BenchmarkOutput, CaseMetadata


@dataclass
class _TaskResult:
    name: str
    samples: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.samples


def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimate the probability that at least one of k samples passes."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0 <= c <= n:
        raise ValueError("c must be between 0 and n")
    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and n")

    if n - c < k:
        return 1.0

    return 1.0 - prod(
        1.0 - k / denominator
        for denominator in range(n - c + 1, n + 1)
    )


def _task_results(
    report: EvaluationReport[str, BenchmarkOutput, CaseMetadata],
) -> list[_TaskResult]:
    results_by_name: dict[str, _TaskResult] = {}

    for case in report.cases:
        task_name = case.source_case_name or case.name
        result = results_by_name.setdefault(task_name, _TaskResult(task_name))
        result.samples += 1

        if not case.evaluator_failures and case.assertions and all(
            assertion.value for assertion in case.assertions.values()
        ):
            result.correct += 1

    for failure in report.failures:
        task_name = failure.source_case_name or failure.name
        result = results_by_name.setdefault(task_name, _TaskResult(task_name))
        result.samples += 1

    return list(results_by_name.values())


@dataclass
class PerTaskAccuracyEvaluator(
    ReportEvaluator[str, BenchmarkOutput, CaseMetadata]
):
    """Report each task's accuracy across all of its repeats."""

    def evaluate(
        self,
        ctx: ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
    ) -> TableResult:
        return TableResult(
            title="Per-task accuracy",
            description=(
                "Fraction of repeats where all assertions for the task passed."
            ),
            columns=["Task", "Correct", "Samples", "Accuracy"],
            rows=[
                [result.name, result.correct, result.samples, result.accuracy]
                for result in _task_results(ctx.report)
            ],
        )


@dataclass
class PerTaskPassAtKEvaluator(
    ReportEvaluator[str, BenchmarkOutput, CaseMetadata]
):
    """Report pass@k for every task and every k supported by its sample count."""

    def evaluate(
        self,
        ctx: ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
    ) -> TableResult:
        rows: list[list[str | int | float | bool | None]] = []
        for result in _task_results(ctx.report):
            for k in range(1, result.samples + 1):
                rows.append(
                    [
                        result.name,
                        k,
                        result.correct,
                        result.samples,
                        pass_at_k(result.samples, result.correct, k),
                    ]
                )

        return TableResult(
            title="Per-task pass@k",
            description=(
                "Estimated probability that at least one of k samples passes."
            ),
            columns=["Task", "k", "Correct", "Samples", "Pass@k"],
            rows=rows,
        )


@dataclass
class AverageAccuracyEvaluator(
    ReportEvaluator[str, BenchmarkOutput, CaseMetadata]
):
    """Report the macro-average of the per-task accuracies."""

    def evaluate(
        self,
        ctx: ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
    ) -> ScalarResult:
        results = _task_results(ctx.report)
        value = (
            sum(result.accuracy for result in results) / len(results)
            if results
            else 0.0
        )
        return ScalarResult(
            title="Average accuracy",
            description="Macro-average of accuracy across tasks.",
            value=value,
        )


@dataclass
class AveragePassAtKEvaluator(
    ReportEvaluator[str, BenchmarkOutput, CaseMetadata]
):
    """Report the macro-average of per-task pass@k for each supported k."""

    def evaluate(
        self,
        ctx: ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
    ) -> TableResult:
        results = _task_results(ctx.report)
        maximum_k = max((result.samples for result in results), default=0)
        rows: list[list[str | int | float | bool | None]] = []

        for k in range(1, maximum_k + 1):
            values = [
                pass_at_k(result.samples, result.correct, k)
                for result in results
                if k <= result.samples
            ]
            rows.append([k, len(values), sum(values) / len(values)])

        return TableResult(
            title="Average pass@k",
            description=(
                "Macro-average of per-task pass@k over tasks with at least k "
                "samples."
            ),
            columns=["k", "Tasks", "Average Pass@k"],
            rows=rows,
        )


DEFAULT_REPORT_EVALUATORS = (
    PerTaskAccuracyEvaluator,
    PerTaskPassAtKEvaluator,
    AverageAccuracyEvaluator,
    AveragePassAtKEvaluator,
)
