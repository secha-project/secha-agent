import json
from pathlib import Path

import pytest

from scripts.bootstrap_pass_rate import (
    TaskScore,
    bootstrap_pass_rate,
    bootstrap_pass_rate_difference,
    load_task_scores,
)


def test_load_task_scores(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "results": {
                    "overall": {},
                    "categories": {
                        "group-1": {
                            "summary": {"tasks": 1, "pass_at_1": 2 / 3},
                            "tasks": {
                                "case-a": {
                                    "name": "task-a",
                                    "summary": {
                                        "correct": 2,
                                        "attempts": 3,
                                        "pass_at_1": 2 / 3,
                                    },
                                    "attempts": [],
                                }
                            },
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    assert load_task_scores(report_path) == [
        TaskScore("case-a", "task-a", "group-1", 2 / 3)
    ]


def test_single_report_uses_equal_category_weights_and_common_m() -> None:
    scores = [
        TaskScore("a", "task-a", "large", 1.0),
        TaskScore("b", "task-b", "large", 1.0),
        TaskScore("c", "task-c", "small", 0.0),
    ]

    result = bootstrap_pass_rate(
        scores,
        tasks_per_category=5,
        bootstrap_samples=100,
        seed=42,
    )

    assert result.estimate == 0.5
    assert result.confidence_interval == (0.5, 0.5)
    assert result.tasks_per_category == 5
    assert result.categories == 2


def test_two_reports_use_paired_differences_and_common_m() -> None:
    scores_a = [
        TaskScore("a", "task-a", "large", 1.0),
        TaskScore("b", "task-b", "large", 1.0),
        TaskScore("c", "task-c", "small", 0.0),
    ]
    scores_b = [
        TaskScore("a", "task-a", "large", 0.0),
        TaskScore("b", "task-b", "large", 0.0),
        TaskScore("c", "task-c", "small", 0.0),
    ]

    result = bootstrap_pass_rate_difference(
        scores_a,
        scores_b,
        tasks_per_category=5,
        bootstrap_samples=100,
        seed=42,
    )

    assert result.estimate == 0.5
    assert result.confidence_interval == (0.5, 0.5)


def test_paired_bootstrap_requires_identical_case_ids() -> None:
    with pytest.raises(ValueError, match="identical case IDs"):
        bootstrap_pass_rate_difference(
            [TaskScore("a", "task-a", "group", 1.0)],
            [TaskScore("b", "task-b", "group", 1.0)],
            tasks_per_category=1,
        )
