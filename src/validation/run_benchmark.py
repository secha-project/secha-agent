import argparse
from pathlib import Path

from src.validation.config import load_run_config
from src.validation.runner import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configured benchmark")
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Run config name from src/validation/run_configs "
            "(for example, 'openai-mini')"
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume an interrupted run from its benchmark-results directory.",
    )
    parser.add_argument(
        "--redo",
        nargs="+",
        default=(),
        metavar="TASK",
        help=(
            "Redo the named benchmark tasks (including all repetitions). "
            "Requires --resume."
        ),
    )
    args = parser.parse_args()
    if args.redo and args.resume is None:
        parser.error("--redo requires --resume")
    return args


def main() -> None:
    args = parse_args()
    run_name, run_config = load_run_config(args.config)
    run_benchmark(
        run_name,
        run_config,
        resume_directory=args.resume,
        redo_tasks=args.redo,
    )


if __name__ == "__main__":
    main()
