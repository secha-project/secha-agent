from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
from pydantic import TypeAdapter
from pydantic_evals.reporting import (
    EvaluationReport,
    ReportCase,
    ReportCaseFailure,
)

from src.validation.eval_models import BenchmarkOutput, CaseMetadata

BenchmarkResult = (
    ReportCase[str, BenchmarkOutput, CaseMetadata]
    | ReportCaseFailure[str, BenchmarkOutput, CaseMetadata]
)
BenchmarkReport = EvaluationReport[str, BenchmarkOutput, CaseMetadata]
_BENCHMARK_REPORT_ADAPTER = TypeAdapter(BenchmarkReport)


class EvaluationCheckpoint:
    """Persist completed benchmark attempts so interrupted runs can resume."""

    def __init__(
        self,
        path: Path,
        *,
        name: str,
        experiment_metadata: dict[str, Any],
    ) -> None:
        self.path = path
        self._lock = anyio.Lock()
        if path.exists():
            report = _BENCHMARK_REPORT_ADAPTER.validate_json(path.read_bytes())
            if report.name != name:
                raise ValueError(
                    f"Checkpoint {path} belongs to run {report.name!r}, not {name!r}."
                )
            if report.experiment_metadata != experiment_metadata:
                raise ValueError(
                    f"Checkpoint {path} was created with a different run configuration."
                )
            self.report = report
        else:
            self.report = EvaluationReport(
                name=name,
                cases=[],
                failures=[],
                experiment_metadata=experiment_metadata,
            )

    @property
    def completed_names(self) -> set[str]:
        return {
            *(case.name for case in self.report.cases),
            *(failure.name for failure in self.report.failures),
        }

    async def record(self, result: BenchmarkResult) -> None:
        """Record one completed attempt and atomically update the checkpoint."""
        async with self._lock:
            if result.name in self.completed_names:
                return
            if isinstance(result, ReportCase):
                self.report.cases.append(result)
            else:
                self.report.failures.append(result)
            self._write()

    def invalidate(self, attempt_names: set[str]) -> list[BenchmarkResult]:
        """Remove selected attempts so they can be evaluated again."""
        invalidated: list[BenchmarkResult] = [
            result
            for result in [*self.report.cases, *self.report.failures]
            if result.name in attempt_names
        ]
        if not invalidated:
            return []

        self.report.cases = [
            case for case in self.report.cases if case.name not in attempt_names
        ]
        self.report.failures = [
            failure
            for failure in self.report.failures
            if failure.name not in attempt_names
        ]
        self._write()
        return invalidated

    def relocate_artifacts(self, source_dir: Path, destination_dir: Path) -> None:
        """Rebase checkpoint artifact paths after copying a run directory."""
        source_dir = source_dir.resolve()
        destination_dir = destination_dir.resolve()
        changed = False

        for result in [*self.report.cases, *self.report.failures]:
            attributes = getattr(result, "attributes", None)
            if not isinstance(attributes, dict):
                continue
            for key in (
                "artifact_paths",
                "csv_path",
                "visualization_spec_path",
                "visualization_image_path",
            ):
                value = attributes.get(key)
                values = value if isinstance(value, list) else [value]
                relocated: list[object] = []
                value_changed = False
                for item in values:
                    if not isinstance(item, str):
                        relocated.append(item)
                        continue
                    path = Path(item).resolve()
                    if not path.is_relative_to(source_dir):
                        relocated.append(item)
                        continue
                    relocated.append(str(destination_dir / path.relative_to(source_dir)))
                    value_changed = True

                if value_changed:
                    attributes[key] = relocated if isinstance(value, list) else relocated[0]
                    changed = True

        if changed:
            self._write()

    def completed_report(
        self,
        attempt_names: list[str],
        *,
        trace_id: str | None,
        span_id: str | None,
    ) -> BenchmarkReport:
        """Return all checkpointed attempts in deterministic benchmark order."""
        cases = {case.name: case for case in self.report.cases}
        failures = {failure.name: failure for failure in self.report.failures}
        missing = [
            name for name in attempt_names if name not in cases and name not in failures
        ]
        if missing:
            raise RuntimeError(
                "Evaluation finished without checkpointing attempts: "
                + ", ".join(missing)
            )
        self.report.trace_id = trace_id or self.report.trace_id
        self.report.span_id = span_id or self.report.span_id
        self._write()
        return EvaluationReport(
            name=self.report.name,
            cases=[cases[name] for name in attempt_names if name in cases],
            failures=[failures[name] for name in attempt_names if name in failures],
            experiment_metadata=self.report.experiment_metadata,
            trace_id=self.report.trace_id,
            span_id=self.report.span_id,
        )

    def _write(self) -> None:
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_bytes(
            _BENCHMARK_REPORT_ADAPTER.dump_json(self.report, indent=2)
        )
        temporary_path.replace(self.path)
