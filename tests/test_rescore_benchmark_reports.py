import asyncio
from copy import deepcopy

import pytest

from scripts.rescore_benchmark_reports import (
    ADAPTER,
    aggregate,
    attempt_index,
    carry_cost_estimates,
    rescore_document,
)


def test_rescore_uses_real_evaluators_and_existing_aggregation(tmp_path):
    artifacts = tmp_path / "run" / "artifacts"
    artifacts.mkdir(parents=True)
    references = tmp_path / "references"
    references.mkdir()
    (references / "table.csv").write_text(
        "timestamp,pw\n2025-03-01 00:00:00.172,25.118034\n"
    )
    (artifacts / "table.csv").write_text(
        "device_name,measurement_time_utc,power_w,window_min_w\n"
        "meter,2025-03-01T00:00:00.172Z,25.118034,25.118034\n"
    )

    def case(name, case_id, expected, output, assertions, attributes=None):
        return {
            "name": name,
            "source_case_name": name.split(" [")[0],
            "inputs": "Return the requested values",
            "metadata": {"case_id": case_id, "category": "data transformations"},
            "expected_output": expected,
            "output": output,
            "metrics": {"tool_calls": 0},
            "attributes": attributes or {},
            "scores": {},
            "labels": {},
            "assertions": {
                key: {
                    "name": key,
                    "value": value,
                    "reason": None,
                    "source": {"name": key, "arguments": None},
                    "evaluator_version": None,
                }
                for key, value in assertions.items()
            },
            "task_duration": 1.0,
            "total_duration": 1.0,
            "trace_id": "trace",
            "span_id": name,
            "evaluator_failures": [],
        }

    numeric_id = "00000000-0000-0000-0000-000000000001"
    table_id = "00000000-0000-0000-0000-000000000002"
    text_id = "00000000-0000-0000-0000-000000000003"
    evaluation = ADAPTER.validate_python(
        {
            "name": "test",
            "experiment_metadata": {
                "run_config": "test",
                "benchmark": {"repetitions": 2},
            },
            "trace_id": "root-trace",
            "span_id": "root-span",
            "failures": [],
            "cases": [
                case(
                    "numeric [1/2]",
                    numeric_id,
                    [{"value": [1930.1304]}],
                    "1,930.13",
                    {"ValidateNumberOutput": False},
                ),
                case(
                    "numeric [2/2]",
                    numeric_id,
                    [{"value": [1930.1304]}],
                    "1,930.14",
                    {"ValidateNumberOutput": False},
                ),
                case(
                    "table [1/1]",
                    table_id,
                    [{"table_path": "table.csv", "ordered": True}],
                    "A table is attached",
                    {"ValidateTableIncluded": True, "ValidateTableValues": False},
                    {"csv_path": "/original/remote/artifacts/table.csv"},
                ),
                case(
                    "text [1/1]",
                    text_id,
                    [{"text": ["answer"]}],
                    "answer",
                    {"ValidateTextOutput": True},
                ),
            ],
        }
    )
    source = asyncio.run(aggregate(evaluation))
    original = deepcopy(source)
    corrected, _, changes, counts = asyncio.run(
        rescore_document(source, artifacts.parent, references)
    )
    assert source == original
    attempts = attempt_index(corrected)
    assert attempts["numeric [1/2]"][1]["passed"] is True
    assert attempts["numeric [2/2]"][1]["passed"] is False
    assert attempts["table [1/1]"][1]["passed"] is True
    assert attempts["text [1/1]"][1] == attempt_index(source)["text [1/1]"][1]
    assert (
        attempts["table [1/1]"][1]["attributes"]["csv_path"]
        == "/original/remote/artifacts/table.csv"
    )
    category = corrected["results"]["categories"]["data transformations"]
    numeric = category["tasks"][numeric_id]
    assert numeric["summary"]["correct"] == 1
    assert numeric["pass_at_k"] == {"1": 0.5, "2": 1.0}
    assert numeric["pass_power_k"] == {"1": 0.5, "2": 0.0}
    assert counts == {
        "ValidateNumberOutput": 2,
        "ValidateTableIncluded": 1,
        "ValidateTableValues": 1,
    }
    assert len(changes) == 2


def cost_documents():
    original = {
        "experiment_metadata": {"run_config": "glm", "trace_id": "trace"},
        "results": {
            "overall": {
                "usage": {
                    "estimated_cost_usd": None,
                    "pricing_source": "Unavailable",
                    "input_tokens": None,
                    "attempts_with_usage": 0,
                }
            },
            "categories": {
                "category": {
                    "summary": {
                        "estimated_cost_usd": None,
                        "pricing_source": "Unavailable",
                    },
                    "tasks": {
                        "task-id": {
                            "name": "task",
                            "attempts": [
                                {
                                    "name": "task [1/3]",
                                    "span_id": "span",
                                    "trace_id": "trace",
                                    "status": "completed",
                                    "passed": False,
                                    "metrics": {"requests": 2},
                                    "output": "answer",
                                }
                            ],
                        }
                    },
                }
            },
        },
    }
    corrected = deepcopy(original)
    corrected["results"]["categories"]["category"]["tasks"]["task-id"]["attempts"][0][
        "passed"
    ] = True
    estimates = deepcopy(original)
    for node in (
        estimates["results"]["overall"]["usage"],
        estimates["results"]["categories"]["category"]["summary"],
    ):
        node["estimated_cost_usd"] = 4.25
        node["pricing_source"] = "Estimate: calibrated historical cost"
    return corrected, estimates


def test_carry_estimates_preserves_corrected_scores_and_missing_token_measurements():
    corrected, estimates = cost_documents()
    untouched_source = deepcopy(estimates)
    before = deepcopy(corrected)
    assert carry_cost_estimates(corrected, estimates) == ["overall", "category"]
    assert estimates == untouched_source
    usage = corrected["results"]["overall"]["usage"]
    assert usage["estimated_cost_usd"] == 4.25
    assert usage["input_tokens"] is None and usage["attempts_with_usage"] == 0
    category = corrected["results"]["categories"]["category"]
    assert category["summary"]["estimated_cost_usd"] == 4.25
    assert category["tasks"] == before["results"]["categories"]["category"]["tasks"]


@pytest.mark.parametrize(
    "mismatch", ["metadata", "scope", "trace", "metrics", "sum", "label"]
)
def test_carry_estimates_rejects_mismatched_or_unlabeled_sources_before_mutation(
    mismatch,
):
    corrected, estimates = cost_documents()
    attempt = estimates["results"]["categories"]["category"]["tasks"]["task-id"][
        "attempts"
    ][0]
    if mismatch == "metadata":
        estimates["experiment_metadata"]["run_config"] = "other run"
    elif mismatch == "scope":
        attempt["span_id"] = "unsampled-span"
    elif mismatch == "trace":
        attempt["trace_id"] = "other-trace"
    elif mismatch == "metrics":
        attempt["metrics"]["requests"] = 3
    elif mismatch == "sum":
        estimates["results"]["overall"]["usage"]["estimated_cost_usd"] = 8.50
    else:
        estimates["results"]["overall"]["usage"]["pricing_source"] = "Unavailable"
    before = deepcopy(corrected)
    with pytest.raises(ValueError):
        carry_cost_estimates(corrected, estimates)
    assert corrected == before
