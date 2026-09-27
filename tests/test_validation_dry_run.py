from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic_ai import models

import src.validation.run_benchmark as cli
from src.validation.config import load_run_config

RUN_CONFIG = "gpt-6-sol__baseline"


def test_dry_run_validates_inputs_without_external_calls_or_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    forbidden = Mock(side_effect=AssertionError("External setup during dry run"))
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr(cli, "run_benchmark", forbidden)
    monkeypatch.setattr("src.validation.runner.create_run_directory", forbidden)
    monkeypatch.setattr("src.validation.runner.create_dependencies", forbidden)
    monkeypatch.setattr("src.validation.runner.configure_observability", forbidden)

    log = tmp_path / "progress.log"
    cli.main(["--config", RUN_CONFIG, "--dry-run", "--progress-log", str(log)])

    output = capsys.readouterr().out
    assert f"Dry run OK: {RUN_CONFIG}" in output
    assert "reference files checked" in output
    assert "no benchmark results written" in output
    forbidden.assert_not_called()
    assert not log.exists()


def test_cli_passes_progress_log_to_runner(monkeypatch, tmp_path):
    run = Mock()
    monkeypatch.setattr(cli, "run_benchmark", run)
    log = tmp_path / "progress.log"
    cli.main(["--config", RUN_CONFIG, "--progress-log", str(log)])
    assert run.call_args.kwargs["progress_log"] == log


def test_dry_run_rejects_missing_reference_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    name, config = load_run_config(RUN_CONFIG)
    monkeypatch.setattr(cli, "BENCHMARKS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Missing benchmark reference files"):
        cli.dry_run(name, config)


@pytest.mark.parametrize("other", [["--resume", "some-run"], ["--redo", "task_1"]])
def test_dry_run_rejects_resume_and_redo(other: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["--config", RUN_CONFIG, "--dry-run", *other])
    assert exc.value.code == 2
