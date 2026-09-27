from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_evals.evaluators import ReportEvaluatorContext

import src.validation.evaluators.report as report_module
from src.validation.eval_models import BenchmarkOutput, CaseMetadata
from src.validation.evaluators.report import (
    AverageAccuracyEvaluator,
    AveragePassAtKEvaluator,
    AveragePassPowerKEvaluator,
    PerCategoryPassMetricsEvaluator,
    PerCategoryTokenUsageAndCostEvaluator,
    PerTaskPassPowerKEvaluator,
    PerTaskStatisticalSummaryEvaluator,
    StratifiedAveragePassAtOneEvaluator,
    TokenUsageAndCostEvaluator,
    pass_power_k,
)


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [
        (5, 3, 1, 3 / 5),
        (5, 3, 2, 3 / 10),
        (5, 3, 3, 1 / 10),
        (5, 3, 4, 0.0),
        (5, 5, 5, 1.0),
    ],
)
def test_pass_power_k(n: int, c: int, k: int, expected: float) -> None:
    assert pass_power_k(n, c, k) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("n", "c", "k"),
    [(0, 0, 1), (2, -1, 1), (2, 3, 1), (2, 1, 0), (2, 1, 3)],
)
def test_pass_power_k_rejects_invalid_arguments(n: int, c: int, k: int) -> None:
    with pytest.raises(ValueError):
        pass_power_k(n, c, k)


def test_pass_power_k_report_evaluators() -> None:
    passing_assertion = SimpleNamespace(value=True)
    failing_assertion = SimpleNamespace(value=False)
    task_a_metadata = SimpleNamespace(
        case_id="case-a", category=SimpleNamespace(value="group")
    )
    task_b_metadata = SimpleNamespace(
        case_id="case-b", category=SimpleNamespace(value="group")
    )
    report = SimpleNamespace(
        cases=[
            SimpleNamespace(
                name="task-a-1",
                source_case_name="task-a",
                evaluator_failures=[],
                assertions={"correct": passing_assertion},
                metadata=task_a_metadata,
            ),
            SimpleNamespace(
                name="task-a-2",
                source_case_name="task-a",
                evaluator_failures=[],
                assertions={"correct": passing_assertion},
                metadata=task_a_metadata,
            ),
            SimpleNamespace(
                name="task-a-3",
                source_case_name="task-a",
                evaluator_failures=[],
                assertions={"correct": failing_assertion},
                metadata=task_a_metadata,
            ),
            SimpleNamespace(
                name="task-b-1",
                source_case_name="task-b",
                evaluator_failures=[],
                assertions={"correct": passing_assertion},
                metadata=task_b_metadata,
            ),
            SimpleNamespace(
                name="task-b-2",
                source_case_name="task-b",
                evaluator_failures=[],
                assertions={"correct": failing_assertion},
                metadata=task_b_metadata,
            ),
        ],
        failures=[],
    )
    ctx = cast(
        ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
        SimpleNamespace(report=report),
    )

    per_task = PerTaskPassPowerKEvaluator().evaluate(ctx)
    average = AveragePassPowerKEvaluator().evaluate(ctx)

    assert per_task.columns == [
        "Category",
        "Case ID",
        "Task",
        "k",
        "Correct",
        "Samples",
        "Pass^k",
    ]
    assert per_task.rows == [
        ["group", "case-a", "task-a", 1, 2, 3, pytest.approx(2 / 3)],
        ["group", "case-a", "task-a", 2, 2, 3, pytest.approx(1 / 3)],
        ["group", "case-a", "task-a", 3, 2, 3, 0.0],
        ["group", "case-b", "task-b", 1, 1, 2, pytest.approx(1 / 2)],
        ["group", "case-b", "task-b", 2, 1, 2, 0.0],
    ]
    assert average.rows == [
        [1, 2, pytest.approx(7 / 12)],
        [2, 2, pytest.approx(1 / 6)],
        [3, 1, 0.0],
    ]


def test_statistical_summary_and_stratified_pass_at_one() -> None:
    passing_assertion = SimpleNamespace(value=True)
    failing_assertion = SimpleNamespace(value=False)

    def case(
        name: str,
        case_id: str,
        category: str,
        assertion: SimpleNamespace,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            name=name,
            source_case_name=name,
            metadata=SimpleNamespace(
                case_id=case_id,
                category=SimpleNamespace(value=category),
            ),
            evaluator_failures=[],
            assertions={"correct": assertion},
        )

    report = SimpleNamespace(
        cases=[
            case("task-a", "case-a", "group-1", passing_assertion),
            case("task-a", "case-a", "group-1", passing_assertion),
            case("task-b", "case-b", "group-1", failing_assertion),
            case("task-c", "case-c", "group-2", passing_assertion),
        ],
        failures=[],
    )
    ctx = cast(
        ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
        SimpleNamespace(report=report),
    )

    summary = PerTaskStatisticalSummaryEvaluator().evaluate(ctx)
    stratified = StratifiedAveragePassAtOneEvaluator().evaluate(ctx)

    assert summary.columns == [
        "Case ID",
        "Task",
        "Category",
        "Correct",
        "Samples",
        "Pass@1",
    ]
    assert summary.rows == [
        ["case-a", "task-a", "group-1", 2, 2, 1.0],
        ["case-b", "task-b", "group-1", 0, 1, 0.0],
        ["case-c", "task-c", "group-2", 1, 1, 1.0],
    ]
    assert stratified.value == pytest.approx(0.75)


def test_per_category_pass_metrics() -> None:
    passing_assertion = SimpleNamespace(value=True)
    failing_assertion = SimpleNamespace(value=False)

    def case(case_id: str, category: str, passed: bool) -> SimpleNamespace:
        return SimpleNamespace(
            name=case_id,
            source_case_name=case_id,
            metadata=SimpleNamespace(
                case_id=case_id,
                category=SimpleNamespace(value=category),
            ),
            evaluator_failures=[],
            assertions={"correct": passing_assertion if passed else failing_assertion},
        )

    report = SimpleNamespace(
        cases=[
            case("task-a", "group-1", True),
            case("task-a", "group-1", False),
            case("task-b", "group-1", True),
            case("task-b", "group-1", True),
            case("task-c", "group-2", False),
        ],
        failures=[],
    )
    ctx = cast(
        ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
        SimpleNamespace(report=report),
    )

    result = PerCategoryPassMetricsEvaluator().evaluate(ctx)

    assert result.columns == [
        "Category",
        "k",
        "Tasks",
        "Average Pass@k",
        "Average Pass^k",
    ]
    assert result.rows == [
        ["group-1", 1, 2, pytest.approx(0.75), pytest.approx(0.75)],
        ["group-1", 2, 2, 1.0, pytest.approx(0.5)],
        ["group-2", 1, 1, 0.0, 0.0],
    ]
    assert AveragePassAtKEvaluator().evaluate(ctx).rows == [
        [1, 3, pytest.approx(0.375)],
        [2, 2, 1.0],
    ]
    assert AveragePassPowerKEvaluator().evaluate(ctx).rows == [
        [1, 3, pytest.approx(0.375)],
        [2, 2, pytest.approx(0.5)],
    ]


def test_overall_pass_metrics_average_category_means() -> None:
    def case(case_id: str, category: str, passed: bool) -> SimpleNamespace:
        return SimpleNamespace(
            name=case_id,
            source_case_name=case_id,
            metadata=SimpleNamespace(
                case_id=case_id,
                category=SimpleNamespace(value=category),
            ),
            evaluator_failures=[],
            assertions={"correct": SimpleNamespace(value=passed)},
        )

    report = SimpleNamespace(
        cases=[
            case("a", "group-1", True),
            case("a", "group-1", True),
            case("b", "group-1", True),
            case("b", "group-1", True),
            case("c", "group-2", True),
            case("c", "group-2", False),
        ],
        failures=[],
    )
    ctx = cast(
        ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
        SimpleNamespace(report=report),
    )

    category = PerCategoryPassMetricsEvaluator().evaluate(ctx)
    overall_at_k = AveragePassAtKEvaluator().evaluate(ctx)
    overall_power_k = AveragePassPowerKEvaluator().evaluate(ctx)
    accuracy = AverageAccuracyEvaluator().evaluate(ctx)

    assert category.rows == [
        ["group-1", 1, 2, 1.0, 1.0],
        ["group-1", 2, 2, 1.0, 1.0],
        ["group-2", 1, 1, 0.5, 0.5],
        ["group-2", 2, 1, 1.0, 0.0],
    ]
    assert overall_at_k.rows == [[1, 3, 0.75], [2, 3, 1.0]]
    assert overall_power_k.rows == [[1, 3, 0.75], [2, 3, 0.5]]
    assert accuracy.value == 0.75
    assert StratifiedAveragePassAtOneEvaluator().evaluate(ctx).value == accuracy.value


def test_token_usage_and_cost_summary_uses_configured_pricing(
    monkeypatch: Any,
) -> None:
    captured: list[dict[str, Any]] = []

    def fake_calc_price(usage: Any, model_name: str, **kwargs: Any) -> SimpleNamespace:
        captured.append({"usage": usage, "model_name": model_name, "kwargs": kwargs})
        return SimpleNamespace(
            total_price=(usage.input_tokens + usage.output_tokens) / 10_000,
            provider=SimpleNamespace(id="openai"),
            model=SimpleNamespace(id="gpt-5-mini"),
        )

    monkeypatch.setattr(report_module, "calc_price", fake_calc_price)
    report = SimpleNamespace(
        cases=[
            SimpleNamespace(
                name="task-a",
                metadata=SimpleNamespace(category=SimpleNamespace(value="group-1")),
                metrics={
                    "requests": 2,
                    "tool_calls": 3,
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_tokens": 10,
                },
            ),
            SimpleNamespace(
                name="task-b",
                metadata=SimpleNamespace(category=SimpleNamespace(value="group-2")),
                metrics={
                    "requests": 1,
                    "tool_calls": 1,
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "reasoning_tokens": 5,
                },
            ),
        ],
        experiment_metadata={
            "agent": {
                "model_provider": "tuni",
                "provider_settings": {"model_name": "custom-model"},
                "pricing": {
                    "provider": "openai",
                    "model_name": "gpt-5-mini",
                },
            }
        },
    )
    ctx = cast(
        ReportEvaluatorContext[str, BenchmarkOutput, CaseMetadata],
        SimpleNamespace(report=report),
    )

    summary = TokenUsageAndCostEvaluator().evaluate(ctx)
    category_summary = PerCategoryTokenUsageAndCostEvaluator().evaluate(ctx)

    assert summary.rows[:7] == [
        ["Requests", 3, "requests"],
        ["Tool calls", 4, "calls"],
        ["Input tokens", 150, "tokens"],
        ["Output tokens", 30, "tokens"],
        ["Total tokens", 180, "tokens"],
        ["Cache Read Tokens", 10, "tokens"],
        ["Reasoning Tokens", 5, "tokens"],
    ]
    assert summary.rows[7][0] == "Estimated cost"
    assert summary.rows[7][1] == pytest.approx(0.018)
    assert summary.rows[7][2] == "USD"
    assert summary.rows[8:] == [
        ["Pricing source", "openai:gpt-5-mini", None],
        ["Attempts with usage", 2, "attempts"],
    ]
    assert category_summary.rows == [
        [
            "group-1",
            1,
            2,
            3,
            100,
            20,
            120,
            pytest.approx(0.012),
            "openai:gpt-5-mini",
        ],
        [
            "group-2",
            1,
            1,
            1,
            50,
            10,
            60,
            pytest.approx(0.006),
            "openai:gpt-5-mini",
        ],
    ]
    assert [call["model_name"] for call in captured] == ["gpt-5-mini"] * 4
    assert all(call["kwargs"] == {"provider_id": "openai"} for call in captured)
    assert [call["usage"].input_tokens for call in captured] == [100, 50, 100, 50]
    assert [call["usage"].output_tokens for call in captured] == [20, 10, 20, 10]
    assert [call["usage"].cache_read_tokens for call in captured] == [10, 0, 10, 0]


@pytest.mark.parametrize("include_valid_attempt", [False, True])
def test_missing_usage_is_unavailable_instead_of_free(include_valid_attempt):
    cases = [
        SimpleNamespace(
            metrics={"requests": 7, "reasoning_tokens": 225},
            metadata=SimpleNamespace(category=SimpleNamespace(value="group")),
        )
    ]
    if include_valid_attempt:
        cases.append(
            SimpleNamespace(
                metrics={
                    "requests": 1,
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "cost": 0.01,
                },
                metadata=cases[0].metadata,
            )
        )
    ctx = SimpleNamespace(
        report=SimpleNamespace(
            cases=cases,
            failures=[],
            experiment_metadata={
                "agent": {
                    "model_provider": "openai",
                    "provider_settings": {"model_name": "gpt-6-sol"},
                }
            },
        )
    )
    values = {row[0]: row[1] for row in TokenUsageAndCostEvaluator().evaluate(ctx).rows}
    assert values["Estimated cost"] is None
    assert values["Pricing source"] == "Unavailable (missing token usage)"
    assert values["Input tokens"] is None
    assert values["Output tokens"] is None
    assert values["Total tokens"] is None
    assert values["Attempts with usage"] == int(include_valid_attempt)
    category = PerCategoryTokenUsageAndCostEvaluator().evaluate(ctx).rows[0]
    assert category[4:8] == [None, None, None, None]


def test_gpt_sol_cost_includes_cache_writes_without_double_counting_reasoning():
    case = SimpleNamespace(
        metrics={
            "requests": 7,
            "input_tokens": 30648,
            "output_tokens": 904,
            "cache_read_tokens": 24624,
            "cache_write_tokens": 6003,
            "reasoning_tokens": 225,
        }
    )
    cost, source = report_module._estimate_cost([case], ("gpt-6-sol", "openai", False))
    assert cost == pytest.approx(0.0290143)
    assert source == "openai:gpt-6-sol"
