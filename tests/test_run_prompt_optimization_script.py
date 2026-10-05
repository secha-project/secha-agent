import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("gepa")

from src.validation.run_optimization import parse_args


@pytest.fixture
def launcher(tmp_path):
    root = tmp_path / "project with spaces"
    script = root / "scripts/run_prompt_optimization.sh"
    script.parent.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    shutil.copy2(source / "scripts/run_prompt_optimization.sh", script)
    relative = Path(
        "src/validation/optimization_splits/power-quality-monitoring_40-20-40.yaml"
    )
    (root / relative).parent.mkdir(parents=True)
    shutil.copy2(source / relative, root / relative)
    capture = tmp_path / "command.json"
    executable = tmp_path / "bin/uv"
    executable.parent.mkdir()
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['OPT_TEST_CAPTURE']).write_text(json.dumps({'cwd': os.getcwd(), 'argv': sys.argv[1:]}))\n"
        "sys.exit(int(os.environ.get('OPT_TEST_EXIT_CODE', '0')))\n"
    )
    executable.chmod(0o755)

    def run(*args, exit_code=0):
        capture.unlink(missing_ok=True)
        result = subprocess.run(
            ["bash", str(script), *args],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{executable.parent}{os.pathsep}{os.environ['PATH']}",
                "OPT_TEST_CAPTURE": str(capture),
                "OPT_TEST_EXIT_CODE": str(exit_code),
            },
            text=True,
            capture_output=True,
            check=False,
        )
        command = json.loads(capture.read_text()) if capture.exists() else None
        return result, command

    return run, root


def optimizer_args(command):
    argv = command["argv"]
    return argv[argv.index("src.validation.run_optimization") + 1 :]


def test_launcher_runs_with_project_defaults_from_any_directory(launcher):
    run, root = launcher
    result, command = run("--dry-run")
    assert result.returncode == 0
    assert command["cwd"] == str(root)
    assert command["argv"][:9] == [
        "run",
        "--locked",
        "--no-dev",
        "--no-cache",
        "--group",
        "optimization",
        "python",
        "-u",
        "-m",
    ]
    args = parse_args(optimizer_args(command))
    assert args.config == "qwen3.8-27b__semantic_layer"
    assert args.reflection_model == "gpt-6-sol"
    assert args.results_dir == root / "benchmark-results"
    assert (len(args.train), len(args.validation), len(args.test)) == (22, 11, 22)
    assert (args.max_metric_calls, args.reflection_minibatch_size, args.seed) == (
        100,
        3,
        0,
    )
    assert (
        args.model_request_max_attempts,
        args.model_request_retry_budget_seconds,
    ) == (20, 1800)
    assert (args.max_resumes, args.resume_delay_seconds) == (3, 60)
    assert args.dry_run


def test_launcher_overrides_models_storage_budget_and_components(launcher, tmp_path):
    run, _ = launcher
    results = tmp_path / "other drive" / "results"
    log = tmp_path / "separate logs" / "summary.log"
    result, command = run(
        "--config",
        "another-target",
        "--reflection-model",
        "another-reflector",
        "--results-dir",
        str(results),
        "--progress-log",
        str(log),
        "--max-metric-calls",
        "500",
        "--components",
        "query.instructions",
        "query_data.description",
    )
    assert result.returncode == 0
    args = parse_args(optimizer_args(command))
    assert (
        args.config == "another-target" and args.reflection_model == "another-reflector"
    )
    assert args.results_dir == results and args.progress_log == log
    assert args.max_metric_calls == 500
    assert args.components == ["query.instructions", "query_data.description"]


@pytest.mark.parametrize("joined", [False, True])
def test_launcher_resume_omits_new_experiment_defaults(launcher, tmp_path, joined):
    run, _ = launcher
    directory = tmp_path / "large drive" / "saved run"
    flags = [f"--resume={directory}"] if joined else ["--resume", str(directory)]
    result, command = run(*flags, "--max-resumes", "0")
    assert result.returncode == 0
    args = parse_args(optimizer_args(command))
    assert args.resume == directory and args.max_resumes == 0
    for key in (
        "config",
        "reflection_model",
        "reflection_provider",
        "results_dir",
        "split_file",
        "components",
        "max_metric_calls",
        "seed",
    ):
        assert getattr(args, key) is None


def test_launcher_allows_explicit_case_splits(launcher):
    run, _ = launcher
    result, command = run("--train", "a", "b", "--validation", "c", "--test", "d")
    assert result.returncode == 0
    args = parse_args(optimizer_args(command))
    assert args.split_file is None
    assert args.train == ["a", "b"] and args.validation == ["c"] and args.test == ["d"]


def test_launcher_help_is_offline_and_exit_status_is_preserved(launcher):
    run, _ = launcher
    result, command = run("--help")
    assert result.returncode == 0 and command is None
    assert "--results-dir" in result.stdout and "--resume" in result.stdout
    result, _ = run("--dry-run", exit_code=17)
    assert result.returncode == 17
