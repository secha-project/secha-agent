"""Remember one configuration's run directory across launcher interruptions."""

import fcntl
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class RunState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str
    run_directory: Path | None = None
    completed: bool = False

    def save(self, path: Path) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


@contextmanager
def track_run(path: Path, config: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    # The OS releases the lock after a crash; a second launcher cannot run the
    # same configuration concurrently or race the atomic state update.
    with path.with_suffix(".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                f"Another process is running this configuration: {path}"
            ) from exc
        try:
            state = (
                RunState.model_validate_json(path.read_bytes())
                if path.exists()
                else RunState(config=config)
            )
            if state.config != config:
                raise ValueError(
                    f"Run state {path} belongs to {state.config}, not {config}"
                )
            if state.completed and state.run_directory is None:
                raise ValueError(f"Completed run state has no run directory: {path}")
            yield state
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
