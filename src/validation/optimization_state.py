"""Durable operations and compact status for resumable prompt optimization."""

from __future__ import annotations

import fcntl
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any

from src.validation.progress_log import ProgressLog


def write_json(path: Path, value: Any) -> None:
    """Readers must never observe a partially written checkpoint or result."""
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def lock_run(directory: Path):
    """OS locks are released even if an optimizer process is killed."""
    with (directory / ".optimizer.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another optimizer is using {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def operation_directory(root: Path, name: str, request: dict) -> Path:
    directory = root / name
    directory.mkdir(exist_ok=True)
    path = directory / "request.json"
    if path.exists():
        if json.loads(path.read_text()) != request:
            raise ValueError(
                f"Resume operation does not match saved inputs: {directory}"
            )
    else:
        write_json(path, request)
    return directory


class OptimizationServiceError(RuntimeError):
    """A service failure must not become evidence against a prompt candidate."""

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


def model_service_error(message: str) -> bool:
    return message.startswith(
        ("ModelAPIError:", "ModelHTTPError:", "APIConnectionError:", "APITimeoutError:")
    )


def retryable_service_error(message: str) -> bool:
    status = re.search(r"status_code[=:]\s*(\d+)", message)
    return status is None or int(status[1]) in {408, 409, 429, 500, 502, 503, 504}


def clear_service_failures(checkpoint: Path) -> list[str]:
    """Keep completed attempts, but leave failed model/judge calls pending."""
    if not checkpoint.exists():
        return []
    data = json.loads(checkpoint.read_text())
    messages = []
    failures = []
    for failure in data["failures"]:
        if model_service_error(failure["error_message"]):
            messages.append(failure["error_message"])
        else:
            failures.append(failure)
    cases = []
    for case in data["cases"]:
        errors = [
            failure["error_message"]
            for failure in case.get("evaluator_failures", [])
            if failure.get("error_type")
            in {
                "ModelAPIError",
                "ModelHTTPError",
                "APIConnectionError",
                "APITimeoutError",
            }
            or model_service_error(failure["error_message"])
        ]
        if errors:
            messages.extend(errors)
        else:
            cases.append(case)
    if messages:
        data["cases"], data["failures"] = cases, failures
        write_json(checkpoint, data)
    return messages


class OptimizationProgressLog(ProgressLog):
    """The existing web viewer can serve this ordinary, bounded-detail log."""

    def __init__(
        self, path: Path, run_name: str, directory: Path, budget: int, **kwargs
    ):
        super().__init__(path, run_name, directory, budget, **kwargs)
        self._state_lock = Lock()
        self.state: dict[str, Any] = {
            "phase": "initializing",
            "metric_calls": 0,
            "iteration": 0,
        }

    def update(self, **values) -> None:
        with self._state_lock:
            self.state.update(values)

    def event(self, status: str, detail: str = "") -> None:
        self._record(status, detail)

    def _progress(self) -> str:
        with self._state_lock:
            state = dict(self.state)
        batch = state.pop("batch", None)
        total = state.pop("batch_total", 0)
        fields = " ".join(f"{key}={value}" for key, value in state.items())
        fields += f" budget={self.total}"
        if batch is not None:
            try:
                data = json.loads((Path(batch) / "checkpoint.json").read_text())
                errors = len(data["failures"])
                completed = len(data["cases"]) + errors
            except FileNotFoundError:
                completed = errors = 0
            except (OSError, ValueError):
                return fields + " batch_progress=unavailable"
            fields += f" batch={Path(batch).name} completed={completed}/{total} execution_errors={errors}"
        return fields

    def on_iteration_start(self, event: dict) -> None:
        state = event["state"]
        self.update(
            iteration=event["iteration"],
            metric_calls=state.total_num_evals,
            best_validation=max(state.program_full_scores_val_set),
        )

    def on_budget_updated(self, event: dict) -> None:
        self.update(metric_calls=event["metric_calls_used"])

    def on_valset_evaluated(self, event: dict) -> None:
        if event["is_best_program"]:
            self.update(best_validation=round(event["average_score"], 6))
        self.event(
            "VALIDATION",
            f"candidate={event['candidate_idx']} score={event['average_score']:.4f}",
        )
