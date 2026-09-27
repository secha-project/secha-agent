"""Check batch selection and failure handling without paid benchmark calls."""

import json
import os
import shutil
import subprocess
import sys
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
    assert "[1/2][beta] model output" in result.stdout
    assert "[2/2][gamma] retry warning" in result.stdout
    assert "[2/2] DONE: gamma" in result.stdout
    logs = sorted(root.glob("benchmark-results/batch-logs/*/*.log"))
    assert [path.name for path in logs] == ["01_beta.log", "02_gamma.log"]
    assert "[1/2][beta] retry warning" in logs[0].read_text()
    assert all("-u" in call and "--dry-run" not in call for call in calls)


def test_failed_command_stops_queue_and_preserves_exit_code(batch_workspace):
    _, run = batch_workspace
    result, calls = run(fail="beta")
    assert result.returncode == 23
    assert config_names(calls) == ["alpha", "beta"]
    assert "[2/3] FAILED: beta (exit 23)" in result.stderr


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
    assert "[1/1] VALIDATING: beta" in result.stdout
    assert not (root / "benchmark-results").exists()


def test_skipping_every_config_is_a_noop(batch_workspace):
    root, run = batch_workspace
    result, calls = run("--skip", "alpha", "--skip", "beta", "--skip", "gamma")
    assert result.returncode == 0, result.stderr
    assert "No configurations remain" in result.stdout
    assert calls == []
    assert not (root / "benchmark-results").exists()
