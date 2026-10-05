import argparse
from pathlib import Path

from src.agent.agent import create_agent
from src.validation.config import BENCHMARKS_DIR, BenchmarkRunConfig, load_run_config
from src.validation.eval_models import (
    BenchmarkDataset,
    TableOutput,
    VisualizationOutput,
)
from src.validation.runner import load_dataset, run_benchmark


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configured benchmark")
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Run config name from src/validation/run_configs "
            "(for example, 'gpt-6-sol__baseline')"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        type=Path,
        help="Resume an interrupted run from its benchmark-results directory.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration, prompts, dataset, and files without external calls.",
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        help="Save run status and checkpoint progress once per minute to this file.",
    )
    redo = parser.add_mutually_exclusive_group()
    redo.add_argument(
        "--redo",
        nargs="+",
        default=(),
        metavar="TASK",
        help=(
            "Redo the named benchmark tasks (including all repetitions). "
            "Requires --resume."
        ),
    )
    redo.add_argument(
        "--redo-tables",
        action="store_true",
        help=(
            "Redo every task with an expected table output, including all repetitions. "
            "Requires --resume, except when checking the selection with --dry-run."
        ),
    )
    args = parser.parse_args(argv)
    if args.redo and args.resume is None:
        parser.error("--redo requires --resume")
    if args.redo_tables and args.resume is None and not args.dry_run:
        parser.error("--redo-tables requires --resume or --dry-run")
    return args


def table_task_names(dataset: BenchmarkDataset) -> list[str]:
    return [
        case.name
        for case in dataset.cases
        if case.name is not None
        and isinstance(case.expected_output, list)
        and any(isinstance(item, TableOutput) for item in case.expected_output)
    ]


def dry_run(
    run_name: str,
    run_config: BenchmarkRunConfig,
    *,
    redo_tables: bool = False,
) -> None:
    """Validate local inputs without evaluating tasks or creating run artifacts."""
    provider = run_config.agent.load_provider()
    create_agent(run_config.agent.enabled_capabilities)
    dataset = load_dataset(run_config.dataset)
    if not dataset.cases:
        raise ValueError("The benchmark dataset must contain tasks")
    names = [case.name for case in dataset.cases]
    if None in names or len(set(names)) != len(names):
        raise ValueError("Benchmark tasks must have unique names")

    base = BENCHMARKS_DIR / run_config.dataset
    paths: set[Path] = set()
    for case in dataset.cases:
        if case.metadata is not None:
            for value in (case.metadata.query_path, case.metadata.ggsql_path):
                if value:
                    paths.add(base / value)
        if not isinstance(case.expected_output, list):
            continue
        for item in case.expected_output:
            if isinstance(item, TableOutput):
                paths.add(base / item.table_path)
            elif isinstance(item, VisualizationOutput):
                paths.add(base / item.image_path)
                if item.spec_path:
                    paths.add(base / item.spec_path)
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Missing benchmark reference files: {missing}")

    model_name = run_config.agent.model_name or provider.model_name
    print(f"Dry run OK: {run_name}")
    print(f"Model: {run_config.agent.model_provider}/{model_name}")
    print(f"Capabilities: {', '.join(sorted(run_config.agent.enabled_capabilities))}")
    print(
        f"Dataset: {run_config.dataset}; {len(dataset.cases)} tasks x "
        f"{run_config.repetitions} repetitions = "
        f"{len(dataset.cases) * run_config.repetitions} attempts; "
        f"{len(paths)} reference files checked"
    )
    if redo_tables:
        tasks = table_task_names(dataset)
        if not tasks:
            raise ValueError("The benchmark dataset has no table-output tasks")
        print(
            f"Table redo: {len(tasks)} tasks x {run_config.repetitions} repetitions "
            f"= {len(tasks) * run_config.repetitions} new attempts."
        )
        print(f"Tasks: {', '.join(tasks)}")
    print("No model, database, or evaluator calls; no benchmark results written.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_name, run_config = load_run_config(args.config)
    if args.dry_run:
        dry_run(run_name, run_config, redo_tables=args.redo_tables)
        return
    redo_tasks = args.redo
    if args.redo_tables:
        redo_tasks = table_task_names(load_dataset(run_config.dataset))
        if not redo_tasks:
            raise ValueError("The benchmark dataset has no table-output tasks")
    run_benchmark(
        run_name,
        run_config,
        resume_directory=args.resume,
        redo_tasks=redo_tasks,
        progress_log=args.progress_log,
    )


if __name__ == "__main__":
    main()
