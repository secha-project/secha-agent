import json
from pathlib import Path

import pytest

from scripts.generate_ablation_category_figure import ablation_category_data
from scripts.load_benchmark_runs import BenchmarkRun


def _run(tmp_path: Path, ablation: str, rates: dict[str, list[float]]) -> BenchmarkRun:
    report = {
        "results": {
            "categories": {
                category: {
                    "tasks": {
                        f"{category}-{i}": {
                            "name": f"Task {i}",
                            "summary": {"pass_at_1": rate},
                        }
                        for i, rate in enumerate(values)
                    }
                }
                for category, values in rates.items()
            }
        }
    }
    path = tmp_path / f"{ablation}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return BenchmarkRun(
        path=path,
        name=f"model__{ablation}",
        dataset="dataset",
        model_provider="provider",
        model_name="model",
        enabled_capabilities=frozenset(),
        repetitions=2,
        report=report,
    )


def test_stacks_preserve_overall_scores_with_unequal_category_sizes(
    tmp_path: Path,
) -> None:
    runs = [
        _run(tmp_path, "baseline", {"large": [1.0, 1.0], "small": [0.0]}),
        _run(tmp_path, "few_shot", {"large": [0.5, 0.5], "small": [1.0]}),
    ]
    data = ablation_category_data(runs, tasks_per_category=5, bootstrap_samples=100)
    for ablation, expected in (("baseline", 0.5), ("few_shot", 0.75)):
        rows = data.loc[data["ablation"] == ablation]
        assert rows["contribution"].sum() == pytest.approx(expected)
        assert rows["overall_pass_rate"].tolist() == [expected, expected]
        assert rows["overall_lower"].tolist() == [expected, expected]
        assert rows["overall_upper"].tolist() == [expected, expected]
        assert rows["weight"].tolist() == [0.5, 0.5]
    # The faceted layout / CSV retain raw category accuracy, including zero.
    baseline = data.loc[data["ablation"] == "baseline"].set_index("category")
    assert baseline.loc["large", "pass_rate"] == 1.0
    assert baseline.loc["small", "pass_rate"] == 0.0
    assert baseline.loc["large", "tasks"] == 2
    assert baseline.loc["small", "tasks"] == 1


def test_category_comparison_rejects_incompatible_task_sets(tmp_path: Path) -> None:
    runs = [
        _run(tmp_path, "baseline", {"large": [1.0, 1.0], "small": [0.0]}),
        _run(tmp_path, "few_shot", {"large": [0.5], "small": [1.0]}),
    ]
    with pytest.raises(ValueError, match="same tasks and categories"):
        ablation_category_data(runs, tasks_per_category=5, bootstrap_samples=100)
