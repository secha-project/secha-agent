from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Literal

from genai_prices import calc_price
from pydantic_ai.usage import RequestUsage
from pydantic_evals.evaluators import ReportEvaluator, ReportEvaluatorContext
from pydantic_evals.reporting import EvaluationReport
from pydantic_evals.reporting.analyses import ScalarResult, TableResult

from ..eval_models import BenchmarkOutput, CaseMetadata


@dataclass
class _TaskResult:
    name: str
    case_id: str
    category: str
    samples: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.samples


@dataclass(frozen=True)
class _CategoryPassMetrics:
    category: str
    k: int
    tasks: int
    pass_at_k: float
    pass_power_k: float


def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimate the probability that at least one of k samples passes."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0 <= c <= n:
        raise ValueError("c must be between 0 and n")
    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and n")

    # as shown in Figure 3 of https://arxiv.org/abs/2107.03374
    if n - c < k:
        return 1.0

    return 1.0 - prod(1.0 - k / denominator for denominator in range(n - c + 1, n + 1))


def pass_power_k(n: int, c: int, k: int) -> float:
    """Estimate the probability that all of k samples pass."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0 <= c <= n:
        raise ValueError("c must be between 0 and n")
    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and n")

    # equivalent to comb(c, k) / comb(n, k), without large intermediate ints.
    if c < k:
        return 0.0

    return prod((c - offset) / (n - offset) for offset in range(k))


def _task_results(
    report: EvaluationReport[str, BenchmarkOutput, CaseMetadata],
) -> list[_TaskResult]:
    results_by_case_id: dict[str, _TaskResult] = {}

    for case in report.cases:
        task_name = case.source_case_name or case.name
        if case.metadata is None:
            raise ValueError(f"Task {task_name!r} is missing benchmark metadata")
        case_id = str(case.metadata.case_id)
        result = results_by_case_id.setdefault(
            case_id,
            _TaskResult(task_name, case_id, case.metadata.category.value),
        )
        result.samples += 1

        if (
            not case.evaluator_failures
            and case.assertions
            and all(assertion.value for assertion in case.assertions.values())
        ):
            result.correct += 1

    for failure in report.failures:
        task_name = failure.source_case_name or failure.name
        if failure.metadata is None:
            raise ValueError(f"Task {task_name!r} is missing benchmark metadata")
        case_id = str(failure.metadata.case_id)
        result = results_by_case_id.setdefault(
            case_id,
            _TaskResult(task_name, case_id, failure.metadata.category.value),
        )
        result.samples += 1

    return list(results_by_case_id.values())


def _task_results_by_category(
    report: EvaluationReport[str, BenchmarkOutput, CaseMetadata],
) -> dict[str, list[_TaskResult]]:
    results_by_category: dict[str, list[_TaskResult]] = {}
    for result in _task_results(report):
        results_by_category.setdefault(result.category, []).append(result)
    return results_by_category


def _category_pass_metrics(
    results_by_category: dict[str, list[_TaskResult]],
) -> list[_CategoryPassMetrics]:
    metrics: list[_CategoryPassMetrics] = []
    for category, results in sorted(results_by_category.items()):
        for k in range(1, max(result.samples for result in results) + 1):
            supported = [result for result in results if k <= result.samples]
            metrics.append(
                _CategoryPassMetrics(
                    category=category,
                    k=k,
                    tasks=len(supported),
                    pass_at_k=sum(
                        pass_at_k(result.samples, result.correct, k)
                        for result in supported
                    )
                    / len(supported),
                    pass_power_k=sum(
                        pass_power_k(result.samples, result.correct, k)
                        for result in supported
                    )
                    / len(supported),
                )
            )
    return metrics


def _overall_pass_metrics(
    category_metrics: list[_CategoryPassMetrics],
    metric: Literal["pass_at_k", "pass_power_k"],
) -> list[list[int | float]]:
    by_k: dict[int, list[_CategoryPassMetrics]] = {}
    for category in category_metrics:
        by_k.setdefault(category.k, []).append(category)
    return [
        [
            k,
            sum(category.tasks for category in categories),
            sum(getattr(category, metric) for category in categories) / len(categories),
        ]
        for k, categories in sorted(by_k.items())
    ]


def _agent_metadata(report: EvaluationReport[Any, Any, Any]) -> dict[str, Any]:
    metadata = report.experiment_metadata
    if not isinstance(metadata, dict):
        return {}
    agent_metadata = metadata.get("agent")
    return agent_metadata if isinstance(agent_metadata, dict) else {}


def _pricing_reference(
    report: EvaluationReport[Any, Any, Any],
) -> tuple[str, str | None, bool] | None:
    agent_metadata = _agent_metadata(report)
    configured_pricing = agent_metadata.get("pricing")
    if isinstance(configured_pricing, dict):
        model_name = configured_pricing.get("model_name")
        provider = configured_pricing.get("provider")
        if isinstance(model_name, str) and (
            provider is None or isinstance(provider, str)
        ):
            return model_name, provider, True

    provider_settings = agent_metadata.get("provider_settings")
    if not isinstance(provider_settings, dict):
        return None
    model_name = provider_settings.get("model_name")
    if not isinstance(model_name, str):
        return None

    model_provider = agent_metadata.get("model_provider")
    if model_provider == "tuni" and model_name == "Qwen3.8-27B-AIC":
        # Use a catalog price for the same base model at a public provider.
        return "@cf/qwen/qwen3.8-27b", "cloudflare", True
    if model_provider == "zai":
        # The catalog also has Zhipu's domestic tariff for this model.
        return model_name, "zai", True
    provider = model_provider if model_provider == "openai" else None
    return model_name, provider, False


def _metric_totals(cases: list[Any]) -> dict[str, int | float]:
    totals: dict[str, int | float] = {}
    for case in cases:
        for name, value in case.metrics.items():
            totals[name] = totals.get(name, 0) + value
    return totals


def _estimate_cost(
    cases: list[Any],
    pricing_reference: tuple[str, str | None, bool] | None,
) -> tuple[float | None, str]:
    totals = _metric_totals(cases)
    explicitly_configured = pricing_reference is not None and pricing_reference[2]
    if "cost" in totals and not explicitly_configured:
        return float(totals["cost"]), "Recorded request prices"

    pricing_source = "Unavailable"
    if pricing_reference is not None:
        model_name, provider, _ = pricing_reference
        case_costs: list[float] = []
        price = None
        for case in cases:
            metrics = case.metrics
            usage = RequestUsage(
                input_tokens=int(metrics.get("input_tokens", 0)),
                output_tokens=int(metrics.get("output_tokens", 0)),
                cache_write_tokens=int(metrics.get("cache_write_tokens", 0)),
                cache_read_tokens=int(metrics.get("cache_read_tokens", 0)),
            )
            try:
                price = calc_price(usage, model_name, provider_id=provider)
            except LookupError:
                pricing_source = f"Unavailable ({provider or 'auto'}:{model_name})"
                break
            case_costs.append(float(price.total_price))
        else:
            if price is not None:
                prefix = (
                    "Proxy "
                    if (provider, model_name)
                    == ("cloudflare", "@cf/qwen/qwen3.8-27b")
                    else ""
                )
                pricing_source = f"{prefix}{price.provider.id}:{price.model.id}"
            return sum(case_costs), pricing_source

    if "cost" in totals:
        return float(totals["cost"]), "Recorded request prices"
    return None, pricing_source


@dataclass
class TokenUsageAndCostEvaluator(ReportEvaluator[str, BenchmarkOutput, CaseMetadata]):
    """Summarize token usage and estimate the run's model cost."""

    def evaluate(
        self,
        ctx: ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
    ) -> TableResult:
        cases = list(ctx.report.cases) + list(getattr(ctx.report, "failures", []))
        totals = _metric_totals(cases)
        cases_with_usage = 0
        for case in cases:
            if any(
                name == "requests" or name.endswith("_tokens") for name in case.metrics
            ):
                cases_with_usage += 1

        token_metrics = {
            name: value for name, value in totals.items() if name.endswith("_tokens")
        }
        input_tokens = int(token_metrics.get("input_tokens", 0))
        output_tokens = int(token_metrics.get("output_tokens", 0))
        rows: list[list[str | int | float | bool | None]] = [
            ["Requests", int(totals.get("requests", 0)), "requests"],
            ["Tool calls", int(totals.get("tool_calls", 0)), "calls"],
            ["Input tokens", input_tokens, "tokens"],
            ["Output tokens", output_tokens, "tokens"],
            ["Total tokens", input_tokens + output_tokens, "tokens"],
        ]

        displayed_token_metrics = {"input_tokens", "output_tokens"}
        for name in sorted(token_metrics.keys() - displayed_token_metrics):
            rows.append([name.replace("_", " ").title(), token_metrics[name], "tokens"])

        pricing_reference = _pricing_reference(ctx.report)
        estimated_cost, pricing_source = _estimate_cost(cases, pricing_reference)

        rows.extend(
            [
                ["Estimated cost", estimated_cost, "USD"],
                ["Pricing source", pricing_source, None],
                ["Attempts with usage", cases_with_usage, "attempts"],
            ]
        )
        return TableResult(
            title="Token usage and estimated cost",
            description=(
                "Totals from all attempted case runs. Cost uses the configured pricing "
                "reference when available, otherwise recorded request prices."
            ),
            columns=["Metric", "Value", "Unit"],
            rows=rows,
        )


@dataclass
class PerCategoryTokenUsageAndCostEvaluator(
    ReportEvaluator[str, BenchmarkOutput, CaseMetadata]
):
    """Summarize token usage and cost separately for each task category."""

    def evaluate(
        self,
        ctx: ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
    ) -> TableResult:
        cases_by_category: dict[str, list[Any]] = {}
        for case in [*ctx.report.cases, *getattr(ctx.report, "failures", [])]:
            if case.metadata is None:
                raise ValueError(f"Case {case.name!r} is missing benchmark metadata")
            cases_by_category.setdefault(case.metadata.category.value, []).append(case)

        pricing_reference = _pricing_reference(ctx.report)
        rows: list[list[str | int | float | bool | None]] = []
        for category, cases in sorted(cases_by_category.items()):
            totals = _metric_totals(cases)
            attempts_with_usage = sum(
                any(
                    name == "requests" or name.endswith("_tokens")
                    for name in case.metrics
                )
                for case in cases
            )
            input_tokens = int(totals.get("input_tokens", 0))
            output_tokens = int(totals.get("output_tokens", 0))
            estimated_cost, pricing_source = _estimate_cost(cases, pricing_reference)
            rows.append(
                [
                    category,
                    attempts_with_usage,
                    int(totals.get("requests", 0)),
                    int(totals.get("tool_calls", 0)),
                    input_tokens,
                    output_tokens,
                    input_tokens + output_tokens,
                    estimated_cost,
                    pricing_source,
                ]
            )

        return TableResult(
            title="Per-category token usage and estimated cost",
            description="Token and estimated cost totals grouped by task category.",
            columns=[
                "Category",
                "Attempts With Usage",
                "Requests",
                "Tool Calls",
                "Input Tokens",
                "Output Tokens",
                "Total Tokens",
                "Estimated Cost (USD)",
                "Pricing Source",
            ],
            rows=rows,
        )


@dataclass
class PerTaskStatisticalSummaryEvaluator(
    ReportEvaluator[str, BenchmarkOutput, CaseMetadata]
):
    """Report the task-level inputs needed for statistical analysis."""

    def evaluate(
        self,
        ctx: ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
    ) -> TableResult:
        return TableResult(
            title="Per-task statistical summary",
            description=(
                "Task-level pass@1 values with stable identifiers and categories for "
                "stratified and paired analysis."
            ),
            columns=["Case ID", "Task", "Category", "Correct", "Samples", "Pass@1"],
            rows=[
                [
                    result.case_id,
                    result.name,
                    result.category,
                    result.correct,
                    result.samples,
                    result.accuracy,
                ]
                for result in _task_results(ctx.report)
            ],
        )


@dataclass
class PerTaskAccuracyEvaluator(ReportEvaluator[str, BenchmarkOutput, CaseMetadata]):
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
            columns=["Category", "Case ID", "Task", "Correct", "Samples", "Accuracy"],
            rows=[
                [
                    result.category,
                    result.case_id,
                    result.name,
                    result.correct,
                    result.samples,
                    result.accuracy,
                ]
                for result in _task_results(ctx.report)
            ],
        )


@dataclass
class PerTaskPassAtKEvaluator(ReportEvaluator[str, BenchmarkOutput, CaseMetadata]):
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
                        result.category,
                        result.case_id,
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
            columns=[
                "Category",
                "Case ID",
                "Task",
                "k",
                "Correct",
                "Samples",
                "Pass@k",
            ],
            rows=rows,
        )


@dataclass
class PerTaskPassPowerKEvaluator(ReportEvaluator[str, BenchmarkOutput, CaseMetadata]):
    """Report pass^k for every task and every k supported by its sample count."""

    def evaluate(
        self,
        ctx: ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
    ) -> TableResult:
        rows: list[list[str | int | float | bool | None]] = []
        for result in _task_results(ctx.report):
            for k in range(1, result.samples + 1):
                rows.append(
                    [
                        result.category,
                        result.case_id,
                        result.name,
                        k,
                        result.correct,
                        result.samples,
                        pass_power_k(result.samples, result.correct, k),
                    ]
                )

        return TableResult(
            title="Per-task pass^k",
            description="Estimated probability that all of k samples pass.",
            columns=[
                "Category",
                "Case ID",
                "Task",
                "k",
                "Correct",
                "Samples",
                "Pass^k",
            ],
            rows=rows,
        )


@dataclass
class PerCategoryPassMetricsEvaluator(
    ReportEvaluator[str, BenchmarkOutput, CaseMetadata]
):
    """Report macro-average pass metrics within each task category."""

    def evaluate(
        self,
        ctx: ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
    ) -> TableResult:
        rows: list[list[str | int | float | bool | None]] = [
            [metric.category, metric.k, metric.tasks, metric.pass_at_k, metric.pass_power_k]
            for metric in _category_pass_metrics(_task_results_by_category(ctx.report))
        ]

        return TableResult(
            title="Per-category pass metrics",
            description=(
                "Macro-average pass@k and pass^k within each category. Tasks only "
                "contribute where they have at least k samples."
            ),
            columns=["Category", "k", "Tasks", "Average Pass@k", "Average Pass^k"],
            rows=rows,
        )


@dataclass
class AverageAccuracyEvaluator(ReportEvaluator[str, BenchmarkOutput, CaseMetadata]):
    """Report pass@1 averaged across category means."""

    def evaluate(
        self,
        ctx: ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
    ) -> ScalarResult:
        category_metrics = _category_pass_metrics(_task_results_by_category(ctx.report))
        pass_at_one = [metric.pass_at_k for metric in category_metrics if metric.k == 1]
        value = sum(pass_at_one) / len(pass_at_one) if pass_at_one else 0.0
        return ScalarResult(
            title="Average accuracy",
            description="Mean of category-level pass@1 averages.",
            value=value,
        )


@dataclass
class StratifiedAveragePassAtOneEvaluator(
    ReportEvaluator[str, BenchmarkOutput, CaseMetadata]
):
    """Report pass@1 with every task category weighted equally."""

    def evaluate(
        self,
        ctx: ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
    ) -> ScalarResult:
        results_by_category = _task_results_by_category(ctx.report)

        category_scores = [
            sum(result.accuracy for result in results) / len(results)
            for results in results_by_category.values()
        ]
        value = sum(category_scores) / len(category_scores) if category_scores else 0.0
        return ScalarResult(
            title="Stratified average pass@1",
            description=(
                "Mean of category-level pass@1 averages, weighting each category "
                "equally."
            ),
            value=value,
        )


@dataclass
class AveragePassAtKEvaluator(ReportEvaluator[str, BenchmarkOutput, CaseMetadata]):
    """Report pass@k averaged across category means for each supported k."""

    def evaluate(
        self,
        ctx: ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
    ) -> TableResult:
        rows = _overall_pass_metrics(
            _category_pass_metrics(_task_results_by_category(ctx.report)),
            "pass_at_k",
        )

        return TableResult(
            title="Average pass@k",
            description=(
                "Mean of category-level pass@k averages. Categories contribute "
                "where they have tasks with at least k samples."
            ),
            columns=["k", "Tasks", "Average Pass@k"],
            rows=rows,
        )


@dataclass
class AveragePassPowerKEvaluator(ReportEvaluator[str, BenchmarkOutput, CaseMetadata]):
    """Report pass^k averaged across category means for each supported k."""

    def evaluate(
        self,
        ctx: ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
    ) -> TableResult:
        rows = _overall_pass_metrics(
            _category_pass_metrics(_task_results_by_category(ctx.report)),
            "pass_power_k",
        )

        return TableResult(
            title="Average pass^k",
            description=(
                "Mean of category-level pass^k averages. Categories contribute "
                "where they have tasks with at least k samples."
            ),
            columns=["k", "Tasks", "Average Pass^k"],
            rows=rows,
        )


DEFAULT_REPORT_EVALUATORS = (
    TokenUsageAndCostEvaluator,
    PerCategoryTokenUsageAndCostEvaluator,
    PerTaskStatisticalSummaryEvaluator,
    PerTaskAccuracyEvaluator,
    PerTaskPassAtKEvaluator,
    PerTaskPassPowerKEvaluator,
    PerCategoryPassMetricsEvaluator,
    AverageAccuracyEvaluator,
    StratifiedAveragePassAtOneEvaluator,
    AveragePassAtKEvaluator,
    AveragePassPowerKEvaluator,
)
