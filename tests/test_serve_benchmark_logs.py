"""Check the browser's view of new, interrupted, and resumed benchmark logs."""

import json
import os
from contextlib import closing, contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread

import pytest

from scripts.serve_benchmark_logs import latest_log, make_handler, read_tail
from src.validation.progress_log import ProgressLog


@contextmanager
def log_server(logs, log_file=None):
    with ThreadingHTTPServer(("127.0.0.1", 0), make_handler(log_file, logs)) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()

        def get(route="/log"):
            with closing(
                HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            ) as client:
                client.request("GET", route)
                response = client.getresponse()
                return (
                    response.status,
                    dict(response.getheaders()),
                    response.read().decode(),
                )

        try:
            yield get
        finally:
            server.shutdown()
            thread.join(timeout=2)


def save_queue(logs, directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "queue.txt").write_text("\n".join(names) + "\n")
    (logs / "latest-queue").write_text(str(directory) + "\n")


def test_viewer_follows_saved_queue_even_when_unrelated_logs_are_newer(tmp_path):
    logs = tmp_path / "batch-logs"
    selected = logs / "selected"
    save_queue(logs, selected, ["alpha", "beta"])
    log = selected / "01_alpha.log"
    log.write_text("[1/2] RESUMING: alpha\n")
    unrelated = tmp_path / "dataset/gepa-newer/progress.log"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("unrelated optimizer output")
    os.utime(unrelated, ns=(log.stat().st_mtime_ns + 1_000_000,) * 2)
    assert latest_log(logs) == log
    with log_server(logs) as get:
        status, headers, body = get()
        assert status == 200 and headers["Cache-Control"] == "no-store"
        assert "Configs completed: 0/2" in body
        assert "RESUMING: alpha" in body and "unrelated" not in body


def test_open_server_tracks_failure_resume_and_next_config(tmp_path):
    logs = tmp_path / "batch-logs"
    queue = logs / "saved"
    save_queue(logs, queue, ["alpha", "beta"])
    run = tmp_path / "run"
    run.mkdir()
    (run / "checkpoint.json").write_text(
        json.dumps({"cases": [{}, {}], "failures": []})
    )
    alpha = queue / "01_alpha.log"
    with log_server(logs) as get:
        with pytest.raises(RuntimeError), ProgressLog(alpha, "alpha", run, 165):
            assert "START completed=2/165" in get()[2]
            raise RuntimeError("service unavailable")
        assert "FAILED completed=2/165" in get()[2]

        with ProgressLog(alpha, "alpha", run, 165):
            body = get()[2]
            assert body.count("START completed=2/165") == 2
        (queue / "alpha.done").touch()
        beta = queue / "02_beta.log"
        beta.write_text("[2/2] RUNNING: beta\n")
        body = get()[2]
        assert "Configs completed: 1/2" in body
        assert "RUNNING: beta" in body and "service unavailable" not in body
        with beta.open("a") as stream:
            stream.write("[2/2] DONE: beta\n")
        (queue / "beta.done").touch()
        assert "Configs completed: 2/2" in get()[2]


def test_existing_server_switches_to_new_and_explicitly_resumed_queues(tmp_path):
    logs = tmp_path / "batch-logs"
    older, newer = logs / "older", logs / "newer"
    save_queue(logs, older, ["alpha"])
    old_log = older / "01_alpha.log"
    old_log.write_text("old run")
    with log_server(logs) as get:
        assert "old run" in get()[2]
        save_queue(logs, newer, ["beta"])
        # Do not show an unrelated old run while the new queue initializes.
        assert latest_log(logs) is None
        assert "Waiting for a run" in get()[2]
        new_log = newer / "01_beta.log"
        new_log.write_text("new run")
        assert "new run" in get()[2]
        (logs / "latest-queue").write_text(str(older) + "\n")
        assert "old run" in get()[2]


def test_explicit_log_is_pinned_and_html_is_escaped(tmp_path):
    logs = tmp_path / "batch-logs"
    save_queue(logs, logs / "saved", ["alpha"])
    pinned = tmp_path / "chosen.log"
    pinned.write_text("<script>test</script>\n")
    with log_server(logs, pinned) as get:
        status, headers, body = get("/")
        assert status == 200
        assert headers["Content-Type"] == "text/html; charset=utf-8"
        assert "&lt;script&gt;test&lt;/script&gt;" in body
        assert "Configs completed" not in body
        assert "<script>test</script>" in get("/log")[2]
        assert get("/missing")[0] == 404


def test_waiting_and_missing_selected_queue_do_not_fall_back_to_old_logs(tmp_path):
    logs = tmp_path / "batch-logs"
    logs.mkdir()
    with log_server(logs) as get:
        assert "Waiting for a run" in get()[2]
        (logs / "latest-queue").write_text(str(logs / "missing") + "\n")
        old = logs / "older"
        old.mkdir()
        (old / "status.log").write_text("stale run")
        assert "stale run" not in get()[2]
        (logs / "latest-queue").write_text("\n")
        assert "saved queue path is empty" in get()[2]


def test_log_tail_stays_bounded(tmp_path):
    path = tmp_path / "large.log"
    path.write_text("\n".join(f"{n}: " + "x" * 400 for n in range(1000)))
    content = read_tail(path)
    assert len(content.encode()) <= 65_536
    assert len(content.splitlines()) <= 200
    assert content.endswith("999: " + "x" * 400)
