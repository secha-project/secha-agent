"""Check batch selection and failure handling without paid benchmark calls."""

import json
import os
import select
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


@pytest.fixture
def batch_workspace(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "run_benchmarks.sh"
    shutil.copy2(Path(__file__).parents[1] / "scripts/run_benchmarks.sh", script)
    configs = tmp_path / "src/validation/run_configs"
    configs.mkdir(parents=True)
    for name in ("alpha", "beta", "gamma"):
        (configs / f"{name}.yaml").write_text("repetitions: 3\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    uv = binaries / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "config = args[args.index('--config') + 1]\n"
        "with open(os.environ['BATCH_TEST_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "print('model output', flush=True)\n"
        "print('retry warning', file=sys.stderr, flush=True)\n"
        "sys.exit(23 if config == os.environ.get('BATCH_TEST_FAIL') else 0)\n"
    )
    uv.chmod(0o755)
    env = os.environ | {
        "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
        "BATCH_TEST_CALLS": str(tmp_path / "calls.jsonl"),
    }

    def run(*args, fail=None):
        command_env = env | ({"BATCH_TEST_FAIL": fail} if fail else {})
        result = subprocess.run(
            ["bash", str(script), *args],
            cwd=tmp_path.parent,  # Runner must work outside the project root.
            env=command_env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        calls_file = tmp_path / "calls.jsonl"
        calls = (
            [json.loads(line) for line in calls_file.read_text().splitlines()]
            if calls_file.exists()
            else []
        )
        return result, calls

    return tmp_path, run


def config_names(calls):
    return [call[call.index("--config") + 1] for call in calls]


def test_skip_completed_config_and_label_each_run_and_log(batch_workspace):
    root, run = batch_workspace
    result, calls = run("--skip", "alpha.yaml")
    assert result.returncode == 0, result.stderr
    assert config_names(calls) == ["beta", "gamma"]
    assert "[1/2] RUNNING: beta" in result.stdout
    assert "[2/2] RUNNING: gamma" in result.stdout
    assert result.stdout.count("model output") == 2
    assert result.stderr.count("retry warning") == 2
    assert "[2/2] DONE: gamma" in result.stdout
    logs = sorted(root.glob("benchmark-results/batch-logs/*/*.log"))
    assert [path.name for path in logs] == ["01_beta.log", "02_gamma.log"]
    assert "[1/2] RUNNING: beta" in logs[0].read_text()
    assert "[1/2] DONE: beta" in logs[0].read_text()
    assert "model output" not in logs[0].read_text()
    assert all("--progress-log" in call for call in calls)
    assert all("-u" in call and "--dry-run" not in call for call in calls)


def test_failed_command_stops_queue_and_preserves_exit_code(batch_workspace):
    root, run = batch_workspace
    result, calls = run(fail="beta")
    assert result.returncode == 23
    assert config_names(calls) == ["alpha", "beta"]
    assert "[2/3] FAILED: beta (exit 23)" in result.stderr
    log = next(root.glob("benchmark-results/batch-logs/*/02_beta.log"))
    assert "STOPPED: beta (exit 23)" in log.read_text()


@pytest.mark.parametrize(
    "args", [("--skip", "typo"), ("--skip",), ("--skip", ""), ("--bad",)]
)
def test_invalid_options_never_start_a_benchmark(batch_workspace, args):
    _, run = batch_workspace
    result, calls = run(*args)
    assert result.returncode == 2
    assert calls == []


def test_dry_run_supports_repeated_skips_without_creating_logs(batch_workspace):
    root, run = batch_workspace
    result, calls = run("--skip", "alpha", "--skip", "gamma", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert config_names(calls) == ["beta"]
    assert "--dry-run" in calls[0]
    assert "--progress-log" not in calls[0]
    assert "[1/1] VALIDATING: beta" in result.stdout
    assert not (root / "benchmark-results").exists()


def test_skipping_every_config_is_a_noop(batch_workspace):
    root, run = batch_workspace
    result, calls = run("--skip", "alpha", "--skip", "beta", "--skip", "gamma")
    assert result.returncode == 0, result.stderr
    assert "No configurations remain" in result.stdout
    assert calls == []
    assert not (root / "benchmark-results").exists()


QWEN_SOURCES = {
    "qwen3.8-27b__baseline": "2026-09-28_03-23",
    "qwen3.8-27b__few_shot": "2026-09-28_17-43",
    "qwen3.8-27b__few_shot_semantic_layer": "2026-09-28_10-35",
    "qwen3.8-27b__semantic_layer": "2026-09-29_08-28",
}


def add_qwen_sources(root):
    paths = {}
    for config, timestamp in QWEN_SOURCES.items():
        (root / "src/validation/run_configs" / f"{config}.yaml").write_text(
            "repetitions: 3\n"
        )
        source = (
            root
            / "benchmark-results/power-quality-monitoring"
            / f"{config}_{timestamp}"
        )
        (source / "artifacts").mkdir(parents=True)
        (source / "checkpoint.json").write_text("original checkpoint")
        (source / "artifacts/retained.png").write_bytes(b"original image")
        paths[config] = source
    return paths


@pytest.mark.parametrize("dry_run", [False, True])
def test_qwen_table_redo_selects_four_original_sources_and_preserves_them(
    batch_workspace, dry_run
):
    root, run = batch_workspace
    sources = add_qwen_sources(root)
    before = {
        path: path.read_bytes()
        for source in sources.values()
        for path in source.rglob("*")
        if path.is_file()
    }
    args = ["--redo-qwen-tables", *(["--dry-run"] if dry_run else [])]
    result, calls = run(*args)
    assert result.returncode == 0, result.stderr
    assert config_names(calls) == list(QWEN_SOURCES)
    for call in calls:
        name = call[call.index("--config") + 1]
        assert "--redo-tables" in call
        assert ("--dry-run" in call) == dry_run
        if dry_run:
            assert "--resume" not in call
        else:
            assert call[call.index("--resume") + 1] == str(
                sources[name].relative_to(root)
            )
    assert all(path.read_bytes() == content for path, content in before.items())
    if dry_run:
        assert not (root / "benchmark-results/batch-logs").exists()


def test_qwen_table_redo_rejects_missing_source_before_starting_any_run(
    batch_workspace,
):
    root, run = batch_workspace
    sources = add_qwen_sources(root)
    sources["qwen3.8-27b__semantic_layer"].joinpath("checkpoint.json").unlink()
    result, calls = run("--redo-qwen-tables")
    assert result.returncode == 2
    assert "Missing original checkpoint or artifacts" in result.stderr
    assert calls == []


def test_qwen_table_redo_can_skip_finished_ablations(batch_workspace):
    root, run = batch_workspace
    add_qwen_sources(root)
    result, calls = run("--redo-qwen-tables", "--skip", "qwen3.8-27b__baseline")
    assert result.returncode == 0, result.stderr
    assert config_names(calls) == list(QWEN_SOURCES)[1:]


@pytest.mark.parametrize("interrupt", [False, True])
def test_terminal_shows_rich_progress_before_run_finishes(batch_workspace, interrupt):
    root, _ = batch_workspace
    gate = root / "finish-progress"
    (root / "bin/uv").write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""
            import os
            import time
            from pathlib import Path
            from rich.progress import Progress, TextColumn

            with Progress(
                TextColumn("attempts {task.completed}/{task.total}"),
                auto_refresh=False,
            ) as progress:
                task = progress.add_task("attempts", total=2)
                progress.update(task, advance=1, refresh=True)
                deadline = time.monotonic() + 10
                while not Path(os.environ["BATCH_PROGRESS_GATE"]).exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("Progress was not visible before completion")
                    time.sleep(0.01)
                progress.update(task, advance=1, refresh=True)
            """)
    )
    master, slave = os.openpty()
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"TTY_COMPATIBLE", "TTY_INTERACTIVE", "FORCE_COLOR"}
    } | {
        "PATH": f"{root / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "TERM": "xterm-256color",
        "BATCH_PROGRESS_GATE": str(gate),
    }
    with subprocess.Popen(
        [
            "bash",
            str(root / "scripts/run_benchmarks.sh"),
            "--skip",
            "alpha",
            "--skip",
            "gamma",
        ],
        cwd=root.parent,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
    ) as process:
        os.close(slave)
        output = b""
        try:
            deadline = time.monotonic() + 5
            while b"attempts 1/2" not in output and time.monotonic() < deadline:
                if select.select([master], [], [], 0.1)[0]:
                    output += os.read(master, 65536)
            assert b"attempts 1/2" in output, output.decode(errors="replace")
            assert process.poll() is None  # An in-flight update, not final output.
            if interrupt:
                os.killpg(process.pid, signal.SIGINT)
            else:
                gate.touch()
            assert process.wait(timeout=5) == (130 if interrupt else 0)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            os.close(master)
    log = next(root.glob("benchmark-results/batch-logs/*/*.log")).read_bytes()
    assert b"RUNNING: beta" in log
    assert (b"INTERRUPTED: beta" if interrupt else b"DONE: beta") in log
    assert b"attempts" not in log
    assert b"\x1b" not in log
