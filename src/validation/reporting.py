import json
import re
from pathlib import Path
from typing import Any

from pydantic_evals.reporting import EvaluationReport, EvaluationReportAdapter

from src.validation.eval_models import BenchmarkOutput, CaseMetadata

BenchmarkReport = EvaluationReport[str, BenchmarkOutput, CaseMetadata]


def _readable_analyses(analyses: list[Any]) -> list[Any]:
    """Convert positional table rows to self-describing JSON records."""
    readable: list[Any] = []
    for analysis in analyses:
        if not isinstance(analysis, dict) or analysis.get("type") != "table":
            readable.append(analysis)
            continue

        columns = analysis.get("columns")
        rows = analysis.get("rows")
        if not isinstance(columns, list) or not all(
            isinstance(column, str) for column in columns
        ):
            raise TypeError("Table analysis columns must be a list of strings")
        if not isinstance(rows, list):
            raise TypeError("Table analysis rows must be a list")

        records: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) != len(columns):
                raise ValueError("Table analysis row does not match its columns")
            records.append(dict(zip(columns, row, strict=True)))

        readable.append(
            {
                key: value
                for key, value in analysis.items()
                if key not in {"columns", "rows"}
            }
            | {"records": records}
        )
    return readable


def _snake_case(value: str) -> str:
    value = value.replace("@", "_at_").replace("^", "_power_")
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _attempt_passed(case: dict[str, Any]) -> bool:
    assertions = case.get("assertions")
    return (
        not case.get("evaluator_failures")
        and isinstance(assertions, dict)
        and bool(assertions)
        and all(
            isinstance(assertion, dict) and assertion.get("value") is True
            for assertion in assertions.values()
        )
    )


def _attempt_record(
    case: dict[str, Any],
    *,
    case_id: str,
    failed: bool,
) -> dict[str, Any]:
    trace_id = case.get("trace_id")
    if not isinstance(trace_id, str) or not trace_id:
        raise ValueError(f"Attempt {case.get('name')!r} is missing a trace ID")
    span_id = case.get("span_id")
    if not isinstance(span_id, str) or not span_id:
        raise ValueError(f"Attempt {case.get('name')!r} is missing a span ID")

    record = {
        key: value
        for key, value in case.items()
        if key
        not in {
            "name",
            "source_case_name",
            "metadata",
            "inputs",
            "expected_output",
            "trace_id",
            "span_id",
        }
    }
    return {
        "name": case.get("name"),
        "case_id": case_id,
        "trace_id": trace_id,
        "span_id": span_id,
        "status": "failed" if failed else "completed",
        "passed": False if failed else _attempt_passed(case),
        **record,
    }


def _structured_results(
    analyses: list[Any],
    cases: list[Any] | None = None,
    failures: list[Any] | None = None,
) -> dict[str, Any]:
    """Build an overall/category/task hierarchy from report evaluator analyses."""
    by_title = {
        analysis["title"]: analysis
        for analysis in analyses
        if isinstance(analysis, dict) and isinstance(analysis.get("title"), str)
    }

    def records(title: str) -> list[dict[str, Any]]:
        analysis = by_title.get(title, {})
        values = analysis.get("records", [])
        return [value for value in values if isinstance(value, dict)]

    def scalar(title: str) -> int | float | None:
        value = by_title.get(title, {}).get("value")
        return (
            value
            if isinstance(value, int | float) and not isinstance(value, bool)
            else None
        )

    overall_usage: dict[str, Any] = {}
    for record in records("Token usage and estimated cost"):
        metric = record.get("Metric")
        if not isinstance(metric, str):
            continue
        key = _snake_case(metric)
        if metric == "Estimated cost":
            key = "estimated_cost_usd"
        overall_usage[key] = record.get("Value")

    overall_pass_metrics: dict[str, dict[str, float]] = {
        "pass_at_k": {},
        "pass_power_k": {},
    }
    for record in records("Average pass@k"):
        overall_pass_metrics["pass_at_k"][str(record["k"])] = record["Average Pass@k"]
    for record in records("Average pass^k"):
        overall_pass_metrics["pass_power_k"][str(record["k"])] = record[
            "Average Pass^k"
        ]

    categories: dict[str, dict[str, Any]] = {}

    def category_result(category: str) -> dict[str, Any]:
        return categories.setdefault(
            category,
            {
                "summary": {},
                "pass_at_k": {},
                "pass_power_k": {},
                "tasks": {},
            },
        )

    for record in records("Per-task statistical summary"):
        category = record["Category"]
        case_id = record["Case ID"]
        result = category_result(category)
        result["tasks"][case_id] = {
            "name": record["Task"],
            "summary": {
                "correct": record["Correct"],
                "attempts": record["Samples"],
                "pass_at_1": record["Pass@1"],
            },
            "pass_at_k": {},
            "pass_power_k": {},
            "attempts": [],
        }

    for title, metric, value_column in (
        ("Per-task pass@k", "pass_at_k", "Pass@k"),
        ("Per-task pass^k", "pass_power_k", "Pass^k"),
    ):
        for record in records(title):
            task = category_result(record["Category"])["tasks"].get(record["Case ID"])
            if task is not None:
                task[metric][str(record["k"])] = record[value_column]

    for record in records("Per-category pass metrics"):
        result = category_result(record["Category"])
        k = str(record["k"])
        result["pass_at_k"][k] = record["Average Pass@k"]
        result["pass_power_k"][k] = record["Average Pass^k"]
        if record["k"] == 1:
            result["summary"]["tasks"] = record["Tasks"]
            result["summary"]["pass_at_1"] = record["Average Pass@k"]

    for record in records("Per-category token usage and estimated cost"):
        result = category_result(record["Category"])
        result["summary"].update(
            {
                "attempts_with_usage": record["Attempts With Usage"],
                "requests": record["Requests"],
                "input_tokens": record["Input Tokens"],
                "output_tokens": record["Output Tokens"],
                "total_tokens": record["Total Tokens"],
                "estimated_cost_usd": record["Estimated Cost (USD)"],
                "pricing_source": record["Pricing Source"],
            }
        )

    attempt_span_identifiers: set[tuple[str, str]] = set()
    for case, failed in [
        *((case, False) for case in (cases or [])),
        *((failure, True) for failure in (failures or [])),
    ]:
        if not isinstance(case, dict):
            continue
        metadata = case.get("metadata")
        if not isinstance(metadata, dict):
            continue
        category = metadata.get("category")
        case_id = metadata.get("case_id")
        if not isinstance(category, str) or not isinstance(case_id, str):
            continue
        trace_id = case.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError(f"Attempt {case.get('name')!r} is missing a trace ID")
        span_id = case.get("span_id")
        if not isinstance(span_id, str) or not span_id:
            raise ValueError(f"Attempt {case.get('name')!r} is missing a span ID")
        attempt_span_identifier = (trace_id, span_id)
        if attempt_span_identifier in attempt_span_identifiers:
            raise ValueError(
                "Duplicate attempt span identifier "
                f"(trace_id={trace_id!r}, span_id={span_id!r})"
            )
        attempt_span_identifiers.add(attempt_span_identifier)

        task = category_result(category)["tasks"].get(case_id)
        if task is None:
            continue
        task.setdefault("input", case.get("inputs"))
        task.setdefault("expected_output", case.get("expected_output"))
        task["attempts"].append(
            _attempt_record(case, case_id=case_id, failed=failed)
        )

    for category in categories.values():
        category["summary"]["attempts"] = sum(
            task["summary"]["attempts"] for task in category["tasks"].values()
        )

    known_titles = {
        "Token usage and estimated cost",
        "Per-category token usage and estimated cost",
        "Per-task statistical summary",
        "Per-task accuracy",
        "Per-task pass@k",
        "Per-task pass^k",
        "Per-category pass metrics",
        "Average accuracy",
        "Stratified average pass@1",
        "Average pass@k",
        "Average pass^k",
    }
    additional_analyses = [
        analysis
        for analysis in analyses
        if not isinstance(analysis, dict) or analysis.get("title") not in known_titles
    ]

    overall: dict[str, Any] = {
        "tasks": sum(len(category["tasks"]) for category in categories.values()),
        "attempts": sum(
            category["summary"].get("attempts", 0) for category in categories.values()
        ),
        "average_pass_at_1": scalar("Average accuracy"),
        "stratified_average_pass_at_1": scalar("Stratified average pass@1"),
        **overall_pass_metrics,
        "usage": overall_usage,
    }
    if additional_analyses:
        overall["additional_analyses"] = additional_analyses

    return {
        "overall": overall,
        "categories": categories,
    }


def build_report_document(report: BenchmarkReport) -> dict[str, Any]:
    """Put trace identifiers inside the report's experiment metadata."""
    serialized_report = json.loads(EvaluationReportAdapter.dump_json(report))
    existing_metadata = serialized_report.pop("experiment_metadata") or {}
    trace_id = serialized_report.pop("trace_id")
    span_id = serialized_report.pop("span_id")
    serialized_report.pop("name")
    analyses = _readable_analyses(serialized_report.pop("analyses"))
    cases = serialized_report.pop("cases")
    failures = serialized_report.pop("failures")

    experiment_metadata = {
        "trace_id": trace_id,
        "span_id": span_id,
        **{
            key: value
            for key, value in existing_metadata.items()
            if key not in {"trace_id", "span_id"}
        },
    }
    return {
        "experiment_metadata": experiment_metadata,
        "results": _structured_results(analyses, cases, failures),
        **serialized_report,
    }


def save_report(report: BenchmarkReport, result_path: Path) -> Path:
    report_document = build_report_document(report)
    result_path.write_text(
        json.dumps(report_document, indent=2) + "\n",
        encoding="utf-8",
    )
    return result_path
