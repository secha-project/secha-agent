"""Randomly subsample attempts per task and rebuild benchmark summaries.

Usage:
    python -m scripts.sample_benchmark_report path/to/report.json --seed 0

Writes report-sample.json beside the source by default. Sampling is without
replacement, independently for each task; retained attempt records (including
their original repetition names and trace IDs) are unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from pydantic_evals.evaluators import ReportEvaluatorContext

from src.validation.evaluators.report import DEFAULT_REPORT_EVALUATORS
from src.validation.reporting import BenchmarkReport, build_report_document


def sample_report(
    report: dict[str, Any], *, repetitions: int = 3, seed: int = 0
) -> dict[str, Any]:
    """Return a sampled copy in the original structured report format."""
    source_repetitions = report["experiment_metadata"]["benchmark"]["repetitions"]
    if not 1 <= repetitions <= source_repetitions:
        raise ValueError("repetitions must be between 1 and the source repetitions")
    if report["report_evaluator_failures"]:
        raise ValueError("Cannot sample a report with report evaluator failures")
    if report["results"]["overall"].get("additional_analyses"):
        raise ValueError("Cannot recalculate additional report analyses")

    sampled = deepcopy(report)
    rng = random.Random(seed)
    categories = sampled["results"]["categories"]
    for category_name in sorted(categories):
        category = categories[category_name]
        for case_id in sorted(category["tasks"]):
            task = category["tasks"][case_id]
            attempts = task["attempts"]
            if len(attempts) != source_repetitions:
                raise ValueError(f"Task {case_id!r} has an unexpected attempt count")
            indices = sorted(rng.sample(range(len(attempts)), repetitions))
            task["attempts"] = [attempts[index] for index in indices]

    sampled["experiment_metadata"]["benchmark"]["repetitions"] = repetitions
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    # Preserve the source benchmark order for floating-point aggregation.
    for category_name, category in categories.items():
        for case_id, task in category["tasks"].items():
            for attempt in task["attempts"]:
                case = {
                    **{
                        key: value
                        for key, value in attempt.items()
                        if key not in {"case_id", "status", "passed"}
                    },
                    "source_case_name": task["name"],
                    "metadata": {
                        "case_id": case_id,
                        "category": category_name,
                    },
                    "inputs": task["input"],
                    "expected_output": task["expected_output"],
                }
                if attempt["status"] == "completed":
                    cases.append(case)
                elif attempt["status"] == "failed":
                    failures.append(case)
                else:
                    raise ValueError(f"Unknown attempt status: {attempt['status']!r}")

    metadata = sampled["experiment_metadata"]
    evaluation_report = TypeAdapter(BenchmarkReport).validate_python(
        {
            "name": metadata["run_config"],
            "cases": cases,
            "failures": failures,
            "experiment_metadata": metadata,
            "trace_id": metadata["trace_id"],
            "span_id": metadata["span_id"],
        }
    )
    ctx = ReportEvaluatorContext(
        name=evaluation_report.name,
        report=evaluation_report,
        experiment_metadata=evaluation_report.experiment_metadata,
    )
    for evaluator in DEFAULT_REPORT_EVALUATORS:
        evaluation_report.analyses.append(evaluator().evaluate(ctx))
    results = build_report_document(evaluation_report)["results"]
    sampled["results"]["overall"] = results["overall"]
    for category_name, category in categories.items():
        recalculated = results["categories"][category_name]
        for key in ("summary", "pass_at_k", "pass_power_k"):
            category[key] = recalculated[key]
        for case_id, task in category["tasks"].items():
            for key in ("summary", "pass_at_k", "pass_power_k"):
                task[key] = recalculated["tasks"][case_id][key]
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.report.with_name("report-sample.json")
    if output.resolve() == args.report.resolve():
        parser.error("output must differ from the source report")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    sampled = sample_report(report, repetitions=args.repetitions, seed=args.seed)
    output.write_text(json.dumps(sampled, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {sampled['results']['overall']['attempts']} attempts to {output}")


if __name__ == "__main__":
    main()
