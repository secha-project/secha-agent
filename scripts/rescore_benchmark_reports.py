"""Rerun local number/table evaluators and the unchanged report pipeline.

Run with: python -m scripts.rescore_benchmark_reports
Original report.json/checkpoint.json files and model outputs are preserved.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from pydantic_evals.evaluators import EvaluatorContext, EvaluatorFailure
from pydantic_evals.evaluators._run_evaluator import run_evaluator
from pydantic_evals.otel import SpanTreeRecordingError

from src.validation.eval_models import BenchmarkDataset, NumberOutput, TableOutput
from src.validation.evaluators.number import ValidateNumberOutput
from src.validation.evaluators.report import DEFAULT_REPORT_EVALUATORS
from src.validation.evaluators.table import ValidateTableIncluded, ValidateTableValues
from src.validation.reporting import BenchmarkReport, build_report_document
from src.validation.runner import _apply_report_evaluators

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = TypeAdapter(BenchmarkReport)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backup_file(source: Path, backup_root: Path) -> Path:
    try:
        relative = source.resolve().relative_to(ROOT)
    except ValueError:
        raise ValueError("Backup sources must be inside the repository") from None
    destination = backup_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise ValueError(f"Existing backup differs from original: {source}")
    else:
        shutil.copy2(source, destination)
    return destination


def report_from_document(document: dict[str, Any]) -> BenchmarkReport:
    """Restore precisely the retained attempts, including sampled repetition names.

    This is record conversion, following sample_benchmark_report.py. All metric
    calculations are delegated to the project's existing report evaluators.
    """
    cases = []
    failures = []
    for category_name, category in document["results"]["categories"].items():
        for case_id, task in category["tasks"].items():
            for attempt in task["attempts"]:
                case = {
                    **{
                        key: value
                        for key, value in attempt.items()
                        if key not in {"case_id", "status", "passed"}
                    },
                    "source_case_name": task["name"],
                    "metadata": {"case_id": case_id, "category": category_name},
                    "inputs": task["input"],
                    "expected_output": task["expected_output"],
                }
                if attempt["status"] == "completed":
                    cases.append(case)
                elif attempt["status"] == "failed":
                    failures.append(case)
                else:
                    raise ValueError(f"Unknown status: {attempt['status']!r}")
    metadata = document["experiment_metadata"]
    return ADAPTER.validate_python(
        {
            "name": metadata["run_config"],
            "cases": cases,
            "failures": failures,
            "experiment_metadata": metadata,
            "trace_id": metadata["trace_id"],
            "span_id": metadata["span_id"],
        }
    )


async def aggregate(report: BenchmarkReport) -> dict[str, Any]:
    """Use the same evaluators, runner and serializer as the benchmark."""
    report.analyses.clear()
    report.report_evaluator_failures.clear()
    dataset = BenchmarkDataset(
        name=report.name,
        cases=[],
        report_evaluators=[kind() for kind in DEFAULT_REPORT_EVALUATORS],
    )
    await _apply_report_evaluators(report, dataset)
    return build_report_document(report)


def assert_unchanged_pass_aggregation(original, rebuilt):
    """Before rescoring, confirm that the normal pipeline reproduces old scores."""
    for key in (
        "tasks",
        "attempts",
        "average_pass_at_1",
        "stratified_average_pass_at_1",
        "pass_at_k",
        "pass_power_k",
    ):
        if original["results"]["overall"][key] != rebuilt["results"]["overall"][key]:
            raise ValueError(f"Baseline aggregation differs at overall.{key}")
    for name, category in original["results"]["categories"].items():
        fresh = rebuilt["results"]["categories"][name]
        for key in ("pass_at_k", "pass_power_k"):
            if category[key] != fresh[key]:
                raise ValueError(f"Baseline aggregation differs at {name}.{key}")
        for case_id, task in category["tasks"].items():
            fresh_task = fresh["tasks"][case_id]
            for key in ("summary", "pass_at_k", "pass_power_k"):
                if task[key] != fresh_task[key]:
                    raise ValueError(
                        f"Baseline aggregation differs at {task['name']}.{key}"
                    )


def attempt_index(document):
    return {
        attempt["span_id"]: (task["name"], attempt)
        for category in document["results"]["categories"].values()
        for task in category["tasks"].values()
        for attempt in task["attempts"]
    }


def carry_cost_estimates(corrected, cost_report):
    """Preserve external estimates for exactly the same saved executions.

    Score changes do not change the cost of already executed requests. These
    estimates are not token measurements, so copy only cost and provenance;
    keep the ordinary aggregation results for tokens, coverage, and scores.
    """
    if corrected["experiment_metadata"] != cost_report["experiment_metadata"]:
        raise ValueError("Cost report experiment metadata differs")
    attempts = attempt_index(corrected)
    cost_attempts = attempt_index(cost_report)
    if attempts.keys() != cost_attempts.keys():
        raise ValueError("Cost report does not cover the same retained attempts")
    for span_id, (task_name, attempt) in attempts.items():
        cost_task_name, cost_attempt = cost_attempts[span_id]
        if task_name != cost_task_name:
            raise ValueError(f"Cost report task differs for {span_id}")
        for key in (
            "name",
            "case_id",
            "trace_id",
            "span_id",
            "status",
            "output",
            "metrics",
            "attributes",
            "task_duration",
            "total_duration",
        ):
            if attempt.get(key) != cost_attempt.get(key):
                raise ValueError(f"Cost report execution {key} differs for {span_id}")
    categories = corrected["results"]["categories"]
    cost_categories = cost_report["results"]["categories"]
    if categories.keys() != cost_categories.keys():
        raise ValueError("Cost report categories differ")
    pairs = [
        (
            "overall",
            corrected["results"]["overall"]["usage"],
            cost_report["results"]["overall"]["usage"],
        )
    ]
    for name, category in categories.items():
        pairs.append((name, category["summary"], cost_categories[name]["summary"]))
    for name, _, source in pairs:
        cost = source.get("estimated_cost_usd")
        pricing = source.get("pricing_source")
        if (
            not isinstance(cost, int | float)
            or isinstance(cost, bool)
            or not math.isfinite(cost)
            or cost < 0
            or not isinstance(pricing, str)
            or not pricing.startswith("Estimate:")
        ):
            raise ValueError(f"Cost report lacks a labeled finite estimate for {name}")
    if not math.isclose(
        pairs[0][2]["estimated_cost_usd"],
        sum(source["estimated_cost_usd"] for _, _, source in pairs[1:]),
        rel_tol=0,
        abs_tol=len(categories) * 1e-8,
    ):
        raise ValueError(
            "Cost report category estimates do not sum to its overall estimate"
        )
    for _, destination, source in pairs:
        for key in ("estimated_cost_usd", "pricing_source"):
            destination[key] = source[key]
    return [name for name, _, _ in pairs]


def value_fields(case):
    return {
        kind: {name: result.value for name, result in getattr(case, kind).items()}
        for kind in ("assertions", "labels", "scores")
    }


async def rescore_document(source, run_directory: Path, reference_root: Path):
    report = report_from_document(source)
    baseline = await aggregate(deepcopy(report))
    assert_unchanged_pass_aggregation(source, baseline)
    original = attempt_index(source)
    changes = []
    counts = {
        "ValidateNumberOutput": 0,
        "ValidateTableIncluded": 0,
        "ValidateTableValues": 0,
    }
    for case in report.cases:
        expected = (
            case.expected_output if isinstance(case.expected_output, list) else []
        )
        evaluators = []
        if any(isinstance(item, NumberOutput) for item in expected):
            evaluators.append(ValidateNumberOutput())
        if any(isinstance(item, TableOutput) for item in expected):
            evaluators.extend(
                [
                    ValidateTableIncluded(),
                    ValidateTableValues(artifact_base_path=reference_root),
                ]
            )
        if not evaluators:
            continue
        attributes = deepcopy(case.attributes)
        if (
            any(isinstance(item, TableOutput) for item in expected)
            and "csv_path" in attributes
        ):
            saved = Path(attributes["csv_path"])
            local = run_directory / "artifacts" / saved.name
            if not local.is_file():
                raise FileNotFoundError(f"Saved table missing for {case.name}: {local}")
            attributes["csv_path"] = str(local.resolve())
        ctx = EvaluatorContext(
            name=case.name,
            inputs=case.inputs,
            metadata=case.metadata,
            expected_output=case.expected_output,
            output=case.output,
            duration=case.task_duration,
            attributes=attributes,
            metrics=case.metrics,
            _span_tree=SpanTreeRecordingError(
                "Offline rescore uses saved outputs, not a new task trace"
            ),
        )
        before = value_fields(case)
        for evaluator in evaluators:
            results = await run_evaluator(evaluator, ctx)
            counts[type(evaluator).__name__] += 1
            if isinstance(results, EvaluatorFailure):
                raise RuntimeError(f"{case.name}: {results.error_message}")  # noqa: TRY004
            for result in results:
                destination = (
                    case.assertions
                    if isinstance(result.value, bool)
                    else (case.labels if isinstance(result.value, str) else case.scores)
                )
                destination[result.name] = result
        after = value_fields(case)
        if before != after:
            changes.append(
                {
                    "task": case.source_case_name,
                    "attempt": case.name,
                    "trace_id": case.trace_id,
                    "case_span_id": case.span_id,
                    "before": before,
                    "after": after,
                }
            )
    corrected = await aggregate(report)
    fresh = attempt_index(corrected)
    if original.keys() != fresh.keys():
        raise ValueError("Rescoring changed retained attempt identities")
    for span_id, (_, attempt) in original.items():
        new_attempt = fresh[span_id][1]
        for key in (
            "name",
            "case_id",
            "trace_id",
            "span_id",
            "status",
            "output",
            "metrics",
            "attributes",
            "task_duration",
            "total_duration",
        ):
            if attempt.get(key) != new_attempt.get(key):
                raise ValueError(f"Rescoring changed original {key} for {span_id}")
        if attempt["passed"] and not new_attempt["passed"]:
            raise ValueError(f"Previously passing attempt regressed: {span_id}")
    for change in changes:
        span_id = change["case_span_id"]
        change["passed_before"] = original[span_id][1]["passed"]
        change["passed_after"] = fresh[span_id][1]["passed"]
    return corrected, report, changes, counts


def write_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


async def run(args):
    args.audit_directory.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.runs_root.glob("*/report.json"))
    if not paths:
        raise ValueError("No canonical report.json files found")
    summary = []
    changes = []
    for path in paths:
        old_hash = sha256(path)
        backup = backup_file(path, args.audit_directory / "backups")
        source = json.loads(path.read_text())
        corrected, checkpoint, run_changes, counts = await rescore_document(
            source, path.parent, args.reference_root
        )
        cost_source = path.with_name("report-cost-estimates.json")
        cost_provenance = None
        if cost_source.is_file():
            cost_backup = backup_file(cost_source, args.audit_directory / "backups")
            cost_scope = carry_cost_estimates(
                corrected, json.loads(cost_source.read_text())
            )
            cost_provenance = {
                "source": str(cost_source.relative_to(ROOT)),
                "source_sha256": sha256(cost_source),
                "backup": str(cost_backup.relative_to(ROOT)),
                "copied_cost_scopes": cost_scope,
            }
        output = path.with_name("report-evaluator-fixed.json")
        new_checkpoint = path.with_name("checkpoint-evaluator-fixed.json")
        previous_corrected = None
        output_bytes = (json.dumps(corrected, indent=2) + "\n").encode()
        checkpoint_bytes = ADAPTER.dump_json(checkpoint, indent=2)
        if output.is_file() and output.read_bytes() != output_bytes:
            previous = json.loads(output.read_text())
            previous_attempts = attempt_index(previous)
            next_attempts = attempt_index(corrected)
            if previous_attempts.keys() != next_attempts.keys() or any(
                attempt["passed"] and not next_attempts[span_id][1]["passed"]
                for span_id, (_, attempt) in previous_attempts.items()
            ):
                raise ValueError("Previously corrected attempts regressed")
            previous_backup = backup_file(
                output, args.audit_directory / "before-rescore"
            )
            previous_corrected = {
                "backup": str(previous_backup.relative_to(ROOT)),
                "sha256": sha256(output),
                "passes": sum(a[1]["passed"] for a in previous_attempts.values()),
            }
        if new_checkpoint.is_file() and new_checkpoint.read_bytes() != checkpoint_bytes:
            backup_file(new_checkpoint, args.audit_directory / "before-rescore")
        write_bytes(output, output_bytes)
        write_bytes(new_checkpoint, checkpoint_bytes)
        assert sha256(path) == old_hash == sha256(backup)
        before = attempt_index(source)
        after = attempt_index(corrected)
        entry = {
            "run": path.parent.name,
            "source": str(path.relative_to(ROOT)),
            "source_sha256": old_hash,
            "backup": str(backup.relative_to(ROOT)),
            "corrected_report": str(output.relative_to(ROOT)),
            "corrected_sha256": sha256(output),
            "corrected_checkpoint": str(new_checkpoint.relative_to(ROOT)),
            "attempts": len(before),
            "passes_before": sum(a[1]["passed"] for a in before.values()),
            "passes_after": sum(a[1]["passed"] for a in after.values()),
            "evaluator_runs": counts,
            "changed_verdicts": len(run_changes),
            "category_weighted_pass_before": source["results"]["overall"][
                "average_pass_at_1"
            ],
            "category_weighted_pass_after": corrected["results"]["overall"][
                "average_pass_at_1"
            ],
            "baseline_pass_aggregation_verified": True,
            "external_cost_estimates": cost_provenance,
            "previous_corrected_report": previous_corrected,
        }
        summary.append(entry)
        changes.extend({"run": path.parent.name, **change} for change in run_changes)
        print(
            f"{path.parent.name}: {entry['passes_before']} -> {entry['passes_after']} passes",
            flush=True,
        )
    for name, value in (("summary.json", summary), ("changed-attempts.json", changes)):
        write_bytes(
            args.audit_directory / name, (json.dumps(value, indent=2) + "\n").encode()
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "benchmark-results/power-quality-monitoring",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=ROOT / "src/validation/benchmark/power-quality-monitoring",
    )
    parser.add_argument(
        "--audit-directory",
        type=Path,
        default=ROOT / "output/benchmark-evaluator-fixes-2026-10-05",
    )
    args = parser.parse_args()
    for name in ("runs_root", "reference_root", "audit_directory"):
        setattr(args, name, getattr(args, name).resolve())
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
