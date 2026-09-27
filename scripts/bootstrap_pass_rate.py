"""Bootstrap stratified pass@1 scores from benchmark report JSON files.

Examples:
    python -m scripts.bootstrap_pass_rate single report.json -m 20
    python -m scripts.bootstrap_pass_rate compare model-a.json model-b.json -m 20
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TaskScore:
    case_id: str
    task: str
    category: str
    pass_at_one: float


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    confidence_level: float
    confidence_interval: tuple[float, float]
    bootstrap_samples: int
    tasks_per_category: int
    categories: int
    tasks: int
    seed: int


def load_task_scores(report_path: Path) -> list[TaskScore]:
    """Load task scores from the structured benchmark results."""
    with report_path.open(encoding="utf-8") as report_file:
        report = json.load(report_file)

    if not isinstance(report, dict):
        raise TypeError(f"{report_path} must contain a JSON object")
    results = report.get("results")
    if not isinstance(results, dict):
        raise TypeError(f"{report_path} does not contain structured 'results'")
    categories = results.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ValueError(f"{report_path} has no task scores")

    scores: list[TaskScore] = []
    seen_ids: set[str] = set()
    for category, category_result in categories.items():
        if not isinstance(category, str):
            raise TypeError("Category names must be strings")
        if not category:
            raise ValueError("Category names must be non-empty strings")
        if not isinstance(category_result, dict) or not isinstance(
            category_result.get("tasks"), dict
        ):
            raise TypeError(f"Invalid category result for {category!r}")
        for case_id, task_result in category_result["tasks"].items():
            if not isinstance(case_id, str) or not isinstance(task_result, dict):
                raise TypeError(f"Invalid task in category {category!r}")
            task = task_result.get("name")
            summary = task_result.get("summary")
            if not isinstance(summary, dict):
                raise TypeError(f"Invalid summary for case ID {case_id!r}")
            pass_at_one = summary.get("pass_at_1")
            if not isinstance(task, str) or not task:
                raise ValueError(f"Invalid task name for case ID {case_id!r}")
            if case_id in seen_ids:
                raise ValueError(f"Duplicate case ID {case_id!r} in {report_path}")
            if (
                isinstance(pass_at_one, bool)
                or not isinstance(pass_at_one, (int, float))
                or not 0.0 <= pass_at_one <= 1.0
            ):
                raise ValueError(f"Invalid pass_at_1 for case ID {case_id!r}")

            seen_ids.add(case_id)
            scores.append(TaskScore(case_id, task, category, float(pass_at_one)))

    return scores


def _validate_options(
    tasks_per_category: int,
    bootstrap_samples: int,
    confidence_level: float,
) -> None:
    if tasks_per_category < 1:
        raise ValueError("tasks_per_category must be at least 1")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least 1")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")


def _scores_by_category(scores: Sequence[TaskScore]) -> dict[str, NDArray[np.float64]]:
    grouped: dict[str, list[float]] = {}
    for score in scores:
        grouped.setdefault(score.category, []).append(score.pass_at_one)
    if not grouped:
        raise ValueError("At least one task score is required")
    return {
        category: np.asarray(values, dtype=np.float64)
        for category, values in grouped.items()
    }


def _bootstrap_stratified_mean(
    values_by_category: dict[str, NDArray[np.float64]],
    tasks_per_category: int,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    bootstrap_scores = np.zeros(bootstrap_samples, dtype=np.float64)
    for category in sorted(values_by_category):
        values = values_by_category[category]
        indices = rng.integers(
            0,
            values.size,
            size=(bootstrap_samples, tasks_per_category),
        )
        bootstrap_scores += values[indices].mean(axis=1)
    return bootstrap_scores / len(values_by_category)


def _result(
    estimate: float,
    bootstrap_scores: NDArray[np.float64],
    *,
    confidence_level: float,
    tasks_per_category: int,
    categories: int,
    tasks: int,
    seed: int,
) -> BootstrapResult:
    tail_probability = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(
        bootstrap_scores,
        [tail_probability, 1.0 - tail_probability],
    )
    return BootstrapResult(
        estimate=float(estimate),
        confidence_level=confidence_level,
        confidence_interval=(float(lower), float(upper)),
        bootstrap_samples=int(bootstrap_scores.size),
        tasks_per_category=tasks_per_category,
        categories=categories,
        tasks=tasks,
        seed=seed,
    )


def bootstrap_pass_rate(
    scores: Sequence[TaskScore],
    *,
    tasks_per_category: int,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Calculate stratified pass@1 and its percentile bootstrap interval."""
    _validate_options(tasks_per_category, bootstrap_samples, confidence_level)
    values_by_category = _scores_by_category(scores)
    estimate = np.mean([values.mean() for values in values_by_category.values()])
    bootstrap_scores = _bootstrap_stratified_mean(
        values_by_category,
        tasks_per_category,
        bootstrap_samples,
        np.random.default_rng(seed),
    )
    return _result(
        float(estimate),
        bootstrap_scores,
        confidence_level=confidence_level,
        tasks_per_category=tasks_per_category,
        categories=len(values_by_category),
        tasks=len(scores),
        seed=seed,
    )


def bootstrap_pass_rate_difference(
    scores_a: Sequence[TaskScore],
    scores_b: Sequence[TaskScore],
    *,
    tasks_per_category: int,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Calculate paired stratified pass@1 difference (A - B) and its interval."""
    _validate_options(tasks_per_category, bootstrap_samples, confidence_level)
    by_id_a = {score.case_id: score for score in scores_a}
    by_id_b = {score.case_id: score for score in scores_b}
    if len(by_id_a) != len(scores_a) or len(by_id_b) != len(scores_b):
        raise ValueError("Each report must contain unique case IDs")
    if by_id_a.keys() != by_id_b.keys():
        only_a = sorted(by_id_a.keys() - by_id_b.keys())
        only_b = sorted(by_id_b.keys() - by_id_a.keys())
        raise ValueError(
            "Reports must contain identical case IDs for paired bootstrapping; "
            f"only in A: {only_a}, only in B: {only_b}"
        )

    differences: list[TaskScore] = []
    for case_id in sorted(by_id_a):
        score_a = by_id_a[case_id]
        score_b = by_id_b[case_id]
        if score_a.category != score_b.category:
            raise ValueError(
                f"Category mismatch for case ID {case_id!r}: "
                f"{score_a.category!r} != {score_b.category!r}"
            )
        differences.append(
            TaskScore(
                case_id=case_id,
                task=score_a.task,
                category=score_a.category,
                pass_at_one=score_a.pass_at_one - score_b.pass_at_one,
            )
        )

    values_by_category = _scores_by_category(differences)
    estimate = np.mean([values.mean() for values in values_by_category.values()])
    bootstrap_scores = _bootstrap_stratified_mean(
        values_by_category,
        tasks_per_category,
        bootstrap_samples,
        np.random.default_rng(seed),
    )
    return _result(
        float(estimate),
        bootstrap_scores,
        confidence_level=confidence_level,
        tasks_per_category=tasks_per_category,
        categories=len(values_by_category),
        tasks=len(differences),
        seed=seed,
    )


def _add_bootstrap_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-m",
        "--tasks-per-category",
        type=int,
        required=True,
        help="Number of tasks sampled with replacement from every category.",
    )
    parser.add_argument(
        "-b",
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Number of bootstrap replicates (default: 10000).",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level between 0 and 1 (default: 0.95).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="NumPy random seed (default: 0).",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser(
        "single", help="Bootstrap the stratified pass@1 of one report."
    )
    single.add_argument("report", type=Path)
    _add_bootstrap_options(single)

    compare = subparsers.add_parser(
        "compare", help="Bootstrap the paired pass@1 difference A - B."
    )
    compare.add_argument("report_a", type=Path)
    compare.add_argument("report_b", type=Path)
    _add_bootstrap_options(compare)
    return parser.parse_args()


def _serialize_result(result: BootstrapResult, **extra: Any) -> str:
    return json.dumps({**extra, **asdict(result)}, indent=2)


def main() -> None:
    args = parse_args()
    try:
        if args.command == "single":
            result = bootstrap_pass_rate(
                load_task_scores(args.report),
                tasks_per_category=args.tasks_per_category,
                bootstrap_samples=args.bootstrap_samples,
                confidence_level=args.confidence_level,
                seed=args.seed,
            )
            output = _serialize_result(
                result,
                statistic="stratified_pass_at_1",
                report=str(args.report),
            )
        else:
            result = bootstrap_pass_rate_difference(
                load_task_scores(args.report_a),
                load_task_scores(args.report_b),
                tasks_per_category=args.tasks_per_category,
                bootstrap_samples=args.bootstrap_samples,
                confidence_level=args.confidence_level,
                seed=args.seed,
            )
            output = _serialize_result(
                result,
                statistic="stratified_pass_at_1_difference_a_minus_b",
                report_a=str(args.report_a),
                report_b=str(args.report_b),
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(output)


if __name__ == "__main__":
    main()
