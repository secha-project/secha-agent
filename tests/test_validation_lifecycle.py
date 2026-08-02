import csv
from pathlib import Path
from typing import Any

import src.validation.lifecycle as lifecycle_module
from src.validation.lifecycle import (
    EvaluationArtifactScope,
    write_evaluation_artifacts,
)


def test_evaluation_writer_supports_multiple_artifacts(
    monkeypatch,
    tmp_path: Path,
    make_table_artifact: Any,
) -> None:
    attributes: dict[str, object] = {}
    monkeypatch.setattr(
        lifecycle_module,
        "set_eval_attribute",
        attributes.__setitem__,
    )
    token = lifecycle_module._evaluation_artifact_scope.set(
        EvaluationArtifactScope(directory=tmp_path, basename="task_1")
    )
    try:
        paths = write_evaluation_artifacts(
            (
                make_table_artifact([{"value": 1}], max_rows=None),
                make_table_artifact([{"value": 2}], max_rows=None),
            )
        )
    finally:
        lifecycle_module._evaluation_artifact_scope.reset(token)

    assert paths == (tmp_path / "task_1.csv", tmp_path / "task_1_2.csv")
    assert attributes == {
        "artifact_paths": [str(path) for path in paths],
        "csv_path": str(paths[0]),
    }
    with paths[1].open(newline="") as artifact_file:
        assert list(csv.DictReader(artifact_file)) == [{"value": "2"}]


def test_evaluation_writer_requires_no_scope_when_nothing_was_published() -> None:
    assert write_evaluation_artifacts(()) == ()
