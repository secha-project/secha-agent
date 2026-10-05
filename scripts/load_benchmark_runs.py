"""Load completed benchmark runs from benchmark-results.

Usage: python -m scripts.load_benchmark_runs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "benchmark-results"
USAGE_EVALUATORS = frozenset(
    {"TokenUsageAndCostEvaluator", "PerCategoryTokenUsageAndCostEvaluator"}
)


@dataclass(frozen=True)
class BenchmarkRun:
    path: Path
    name: str
    dataset: str
    model_provider: str
    model_name: str
    enabled_capabilities: frozenset[str]
    repetitions: int
    report: dict[str, Any]

    @property
    def results(self) -> dict[str, Any]:
        return self.report["results"]


def _load_run(path: Path) -> BenchmarkRun | None:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict):
        return None
    evaluator_failures = report.get("report_evaluator_failures")
    # Missing usage analyses do not invalidate completed attempts or their scores.
    # Scoring/unknown evaluator failures still prevent accepting the report.
    if not isinstance(evaluator_failures, list) or any(
        not isinstance(failure, dict)
        or not isinstance(failure.get("name"), str)
        or failure["name"] not in USAGE_EVALUATORS
        for failure in evaluator_failures
    ):
        return None

    metadata = report.get("experiment_metadata")
    if not isinstance(metadata, dict):
        return None
    name = metadata.get("run_config")
    benchmark = metadata.get("benchmark")
    agent = metadata.get("agent")
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(benchmark, dict) or benchmark.get("dataset") != path.parent.parent.name:
        return None
    repetitions = benchmark.get("repetitions")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        return None
    if not isinstance(agent, dict):
        return None
    provider = agent.get("model_provider")
    provider_settings = agent.get("provider_settings")
    capabilities = agent.get("enabled_capabilities")
    if not isinstance(provider, str) or not provider:
        return None
    if not isinstance(provider_settings, dict):
        return None
    model_name = provider_settings.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        return None
    if not isinstance(capabilities, list) or not all(
        isinstance(capability, str) for capability in capabilities
    ):
        return None

    results = report.get("results")
    if not isinstance(results, dict):
        return None
    overall = results.get("overall")
    categories = results.get("categories")
    if not isinstance(overall, dict) or not isinstance(categories, dict) or not categories:
        return None
    tasks: list[dict[str, Any]] = []
    for category in categories.values():
        if not isinstance(category, dict) or not isinstance(category.get("tasks"), dict):
            return None
        for task in category["tasks"].values():
            if not isinstance(task, dict):
                return None
            tasks.append(task)
    if not tasks or overall.get("tasks") != len(tasks):
        return None
    if overall.get("attempts") != len(tasks) * repetitions:
        return None
    if any(
        not isinstance(task.get("attempts"), list)
        or len(task["attempts"]) != repetitions
        for task in tasks
    ):
        return None

    return BenchmarkRun(
        path=path.resolve(),
        name=name,
        dataset=benchmark["dataset"],
        model_provider=provider,
        model_name=model_name,
        enabled_capabilities=frozenset(capabilities),
        repetitions=repetitions,
        report=report,
    )


def load_benchmark_runs(results_dir: Path = DEFAULT_RESULTS_DIR) -> list[BenchmarkRun]:
    """Load complete scored reports, retaining failed attempts and usage errors."""
    runs: list[BenchmarkRun] = []
    for path in sorted(results_dir.glob("*/*/report.json")):
        run = _load_run(path)
        if run is not None:
            runs.append(run)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    runs = load_benchmark_runs(args.results_dir)
    print(
        json.dumps(
            [
                {
                    "path": str(run.path),
                    "name": run.name,
                    "dataset": run.dataset,
                    "model": run.model_name,
                    "capabilities": sorted(run.enabled_capabilities),
                }
                for run in runs
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
