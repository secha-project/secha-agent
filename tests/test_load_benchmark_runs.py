import json
from pathlib import Path

import pytest

from scripts.load_benchmark_runs import load_benchmark_runs


@pytest.mark.parametrize(
    "evaluator_failures",
    [
        [],
        [
            {
                "name": "TokenUsageAndCostEvaluator",
                "error_message": "ReportCaseFailure has no metrics",
            },
            {
                "name": "PerCategoryTokenUsageAndCostEvaluator",
                "error_message": "ReportCaseFailure has no metrics",
            },
        ],
    ],
)
def test_loads_complete_runs_and_retains_failed_attempts(
    tmp_path: Path, evaluator_failures: list[dict[str, str]]
) -> None:
    results_dir = tmp_path / "benchmark-results"
    completed_path = results_dir / "dataset-a" / "model-a_2026-09-22" / "report.json"
    completed_path.parent.mkdir(parents=True)
    completed_report = {
        "experiment_metadata": {
            "run_config": "model-a",
            "benchmark": {"dataset": "dataset-a", "repetitions": 2},
            "agent": {
                "model_provider": "openai",
                "provider_settings": {"model_name": "model-a"},
                "enabled_capabilities": ["catalog", "query"],
            },
        },
        "report_evaluator_failures": evaluator_failures,
        "results": {
            "overall": {"tasks": 1, "attempts": 2},
            "categories": {
                "data discovery": {
                    "tasks": {
                        "case-a": {
                            "attempts": [
                                {"status": "completed", "passed": True},
                                {"status": "failed", "passed": False},
                            ]
                        }
                    }
                }
            },
        },
    }
    completed_path.write_text(json.dumps(completed_report), encoding="utf-8")

    incomplete_path = results_dir / "dataset-a" / "model-b_2026-09-22" / "report.json"
    incomplete_path.parent.mkdir(parents=True)
    incomplete_report = json.loads(json.dumps(completed_report))
    attempts = incomplete_report["results"]["categories"]["data discovery"]["tasks"]["case-a"]["attempts"]
    attempts.pop()
    incomplete_path.write_text(json.dumps(incomplete_report), encoding="utf-8")

    legacy_path = results_dir / "dataset-a" / "old-test-run" / "report.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text('{"cases": []}', encoding="utf-8")

    scoring_failure_path = results_dir / "dataset-a" / "scoring-error" / "report.json"
    scoring_failure_path.parent.mkdir(parents=True)
    scoring_failure_report = json.loads(json.dumps(completed_report))
    scoring_failure_report["report_evaluator_failures"] = [
        {"name": "PerTaskStatisticalSummaryEvaluator", "error_message": "bad scores"}
    ]
    scoring_failure_path.write_text(json.dumps(scoring_failure_report), encoding="utf-8")

    runs = load_benchmark_runs(results_dir)
    assert len(runs) == 1
    assert runs[0].path == completed_path
    assert runs[0].name == "model-a"
    assert runs[0].model_name == "model-a"
    assert runs[0].enabled_capabilities == frozenset({"catalog", "query"})
    assert runs[0].results == completed_report["results"]
