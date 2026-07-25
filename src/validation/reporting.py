import json
from pathlib import Path
from typing import Any

from pydantic_evals.reporting import EvaluationReport, EvaluationReportAdapter

from src.validation.eval_models import BenchmarkOutput, CaseMetadata

BenchmarkReport = EvaluationReport[str, BenchmarkOutput, CaseMetadata]


def build_report_document(report: BenchmarkReport) -> dict[str, Any]:
    """Put trace identifiers inside the report's experiment metadata."""
    serialized_report = json.loads(EvaluationReportAdapter.dump_json(report))
    existing_metadata = serialized_report.pop("experiment_metadata") or {}
    trace_id = serialized_report.pop("trace_id")
    span_id = serialized_report.pop("span_id")
    serialized_report.pop("name")
    analyses = serialized_report.pop("analyses")
    cases = serialized_report.pop("cases")

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
        "analyses": analyses,
        "cases": cases,
        **serialized_report,
    }


def save_report(report: BenchmarkReport, result_path: Path) -> Path:
    report_document = build_report_document(report)
    result_path.write_text(
        json.dumps(report_document, indent=2) + "\n",
        encoding="utf-8",
    )
    return result_path
