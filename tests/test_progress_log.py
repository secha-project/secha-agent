import json
import time

import pytest

from src.validation.progress_log import ProgressLog


def test_logs_periodic_checkpoint_counts_including_resumed_attempts(tmp_path):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    checkpoint = run_directory / "checkpoint.json"
    checkpoint.write_text(json.dumps({"cases": [{}], "failures": [{}]}))
    log = tmp_path / "status.log"

    with ProgressLog(log, "test-config", run_directory, 4, interval_seconds=0.01):
        updated = checkpoint.with_suffix(".tmp")
        updated.write_text(json.dumps({"cases": [{}, {}, {}], "failures": [{}]}))
        updated.replace(checkpoint)
        deadline = time.monotonic() + 2
        while "PROGRESS completed=4/4" not in log.read_text():
            assert time.monotonic() < deadline
            time.sleep(0.005)

    content = log.read_text()
    assert "[test-config] START completed=2/4 execution_errors=1" in content
    assert "PROGRESS completed=4/4 execution_errors=1" in content
    assert "FINISHED completed=4/4 execution_errors=1" in content
    assert f"results={run_directory}" in content


def test_default_log_omits_console_noise_and_frequent_redraws(tmp_path, capsys):
    log = tmp_path / "status.log"
    with ProgressLog(log, "test-config", tmp_path, 165):
        for _ in range(1000):
            print("\r\x1b[2Knoisy progress update", end="")
    assert "noisy progress" in capsys.readouterr().out
    content = log.read_text()
    assert len(content.splitlines()) == 2
    assert "START completed=0/165" in content
    assert "FINISHED completed=0/165" in content
    assert "noisy progress" not in content
    assert "\x1b" not in content
    assert log.stat().st_size < 1000


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (RuntimeError("failure\nsecond line"), "FAILED"),
        (KeyboardInterrupt(), "INTERRUPTED"),
    ],
)
def test_failure_and_interruption_are_logged_and_propagated(tmp_path, error, status):
    log = tmp_path / "status.log"
    with pytest.raises(type(error)), ProgressLog(log, "test-config", tmp_path, 3):
        raise error
    content = log.read_text()
    assert f"{status} completed=0/3" in content
    assert len(content.splitlines()) == 2
    if status == "FAILED":
        assert "RuntimeError: failure second line" in content
