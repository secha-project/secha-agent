import argparse

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name, run_config = load_run_config(args.config)
    run_benchmark(run_name, run_config)


if __name__ == "__main__":
    main()
