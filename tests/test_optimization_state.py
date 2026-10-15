import json
import time
from pathlib import Path

import pytest
import yaml

from src.optimization.state import OptimizationProgressLog, lock_run


def test_optimizer_log_tracks_active_batch_and_finishes(tmp_path):
    batch = tmp_path / "evaluations" / "search-000000"
    batch.mkdir(parents=True)
    (batch / "checkpoint.json").write_text(
        json.dumps({"cases": [{}, {}], "failures": [{}]})
    )
    log = tmp_path / "progress.log"
    with OptimizationProgressLog(
        log, "optimizer", tmp_path, 100, interval_seconds=0.01
    ) as progress:
        progress.update(
            phase="search",
            batch=str(batch),
            batch_total=9,
            metric_calls=5,
            best_validation=0.5,
        )
        deadline = time.monotonic() + 2
        while "completed=3/9" not in log.read_text():
            assert time.monotonic() < deadline
            time.sleep(0.005)
    content = log.read_text()
    assert "metric_calls=5" in content and "budget=100" in content
    assert "execution_errors=1" in content and "best_validation=0.5" in content
    assert "FINISHED" in content


def test_active_optimization_cannot_be_resumed_concurrently(tmp_path):
    with (
        lock_run(tmp_path),
        pytest.raises(ValueError, match="Another optimizer"),
        lock_run(tmp_path),
    ):
        pytest.fail("Must not enter a locked run")
    with lock_run(tmp_path):
        pass


def test_proposed_split_covers_every_case_and_preserves_group_and_stratum_quotas():
    root = Path(__file__).resolve().parents[1] / "src"
    proposal = yaml.safe_load(
        (
            root / "optimization" / "splits" / "power-quality-monitoring_40-20-40.yaml"
        ).read_text()
    )
    dataset = yaml.safe_load(
        (
            root
            / "validation"
            / "benchmark"
            / "power-quality-monitoring"
            / "dataset.yaml"
        ).read_text()
    )
    splits = {key: set(proposal[key]) for key in ("train", "validation", "test")}
    assert [len(value) for value in splits.values()] == [22, 11, 22]
    assert len(set.union(*splits.values())) == 55
    assert set.union(*splits.values()) == {case["name"] for case in dataset["cases"]}
    for group in proposal["groups"]:
        assert sum(set(group) <= split for split in splits.values()) == 1
    for coverage in proposal["coverage"]:
        matching = set()
        for case in dataset["cases"]:
            items = case["expected_output"]
            output = (
                "visualization"
                if any("image_path" in item for item in items)
                else "table"
                if any("table_path" in item for item in items)
                else "numeric_or_mixed"
                if any("value" in item for item in items)
                else "text"
            )
            if (case["metadata"]["category"], output) == (
                coverage["category"],
                coverage["output_family"],
            ):
                matching.add(case["name"])
        for name, split in splits.items():
            assert len(matching & split) == coverage[name]
