"""Rebuild report aggregations from a saved checkpoint without rerunning tasks.

Usage: python -m scripts.aggregate_benchmark_report path/to/report.json

Writes report-complete.json beside the source, preserving its attempt records
and structured report schema.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from pydantic_evals.evaluators import ReportEvaluatorContext

from src.validation.evaluators.report import DEFAULT_REPORT_EVALUATORS
from src.validation.reporting import BenchmarkReport, build_report_document


def aggregate_report(source: dict[str, Any], checkpoint: bytes) -> dict[str, Any]:
    """Run the actual report evaluators and retain original per-attempt data."""
    report = TypeAdapter(BenchmarkReport).validate_json(checkpoint)
    report.analyses.clear()
    report.report_evaluator_failures.clear()
    context = ReportEvaluatorContext(
        name=report.name,
        report=report,
        experiment_metadata=report.experiment_metadata,
    )
    for evaluator_type in DEFAULT_REPORT_EVALUATORS:
        result = evaluator_type().evaluate(context)
        report.analyses.extend(result if isinstance(result, list) else [result])
    rebuilt = build_report_document(report)
    if rebuilt["experiment_metadata"] != source["experiment_metadata"]:
        raise ValueError("Checkpoint and source report metadata do not match")
    original_categories = source["results"]["categories"]
    fresh_categories = rebuilt["results"]["categories"]
    if original_categories.keys() != fresh_categories.keys():
        raise ValueError("Checkpoint and source report categories do not match")
    for category, original in original_categories.items():
        fresh = fresh_categories[category]
        if original["tasks"].keys() != fresh["tasks"].keys():
            raise ValueError(f"Checkpoint tasks differ in category {category!r}")
        for case_id, task in original["tasks"].items():
            old_attempts = {attempt["name"]: attempt for attempt in task["attempts"]}
            new_attempts = {
                attempt["name"]: attempt
                for attempt in fresh["tasks"][case_id]["attempts"]
            }
            if old_attempts != new_attempts:
                raise ValueError(f"Checkpoint attempts differ for task {case_id!r}")
    # Preserve original attempt ordering and any additional report analyses.
    completed = deepcopy(source)
    completed["results"]["overall"].update(rebuilt["results"]["overall"])
    for category, fresh in fresh_categories.items():
        destination = completed["results"]["categories"][category]
        for key in ("summary", "pass_at_k", "pass_power_k"):
            destination[key] = fresh[key]
        for case_id, task in fresh["tasks"].items():
            for key in ("summary", "pass_at_k", "pass_power_k"):
                destination["tasks"][case_id][key] = task[key]
    rerun_names = {evaluator.__name__ for evaluator in DEFAULT_REPORT_EVALUATORS}
    completed["report_evaluator_failures"] = [
        failure for failure in source["report_evaluator_failures"]
        if failure["name"] not in rerun_names
    ]
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint or args.report.with_name("checkpoint.json")
    output = args.output or args.report.with_name("report-complete.json")
    if output.resolve() in {args.report.resolve(), checkpoint.resolve()}:
        parser.error("output must differ from the source report and checkpoint")
    source = json.loads(args.report.read_text(encoding="utf-8"))
    completed = aggregate_report(source, checkpoint.read_bytes())
    output.write_text(json.dumps(completed, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}")
    print(json.dumps(completed["results"]["overall"]["usage"], indent=2))


if __name__ == "__main__":
    main()
