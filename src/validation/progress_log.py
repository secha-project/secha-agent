"""Small periodic status logs, independent of terminal progress rendering."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from types import TracebackType
from typing import Self


class ProgressLog:
    def __init__(
        self,
        path: Path,
        run_name: str,
        run_directory: Path,
        total: int,
        *,
        interval_seconds: float = 60,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("Progress log interval must be positive")
        self.path = path
        self.run_name = run_name
        self.run_directory = run_directory
        self.total = total
        self.interval_seconds = interval_seconds
        self._stop = Event()
        self._thread = Thread(target=self._periodic, daemon=True)
        self._started = time.monotonic()

    def _progress(self) -> str:
        # Checkpoints are replaced atomically, so a reader sees a complete file.
        try:
            checkpoint = json.loads(
                (self.run_directory / "checkpoint.json").read_bytes()
            )
        except FileNotFoundError:
            completed = errors = 0
        except (OSError, ValueError):
            return "progress=unavailable"
        else:
            errors = len(checkpoint.get("failures", []))
            completed = len(checkpoint.get("cases", [])) + errors
        return f"completed={completed}/{self.total} execution_errors={errors}"

    def _record(self, status: str, detail: str = "") -> None:
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        elapsed = int(time.monotonic() - self._started)
        # Keep exceptions on one short line; full tracebacks stay on the console.
        detail = " ".join(detail.split())[:1000]
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"{timestamp} [{self.run_name}] {status} "
                f"{self._progress()} elapsed={elapsed}s {detail}".rstrip()
                + "\n"
            )

    def _periodic(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._record("PROGRESS")

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._record("START", f"results={self.run_directory}")
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join()
        if exc_type is None:
            self._record("FINISHED")
        elif issubclass(exc_type, KeyboardInterrupt):
            self._record("INTERRUPTED")
        else:
            self._record("FAILED", f"{exc_type.__name__}: {exc_value}")
