"""Configuration models and loading helpers for benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = Path(__file__).parent / "benchmark"
RUN_CONFIGS_DIR = Path(__file__).parent / "run_configs"
RESULTS_DIR = PROJECT_ROOT / "benchmark-results"


class BenchmarkRunConfig(BaseModel):
    """Settings for one benchmark run."""

    model_config = ConfigDict(extra="forbid")

    agent: dict[str, Any] = Field(default_factory=dict)
    dataset: str
    repetitions: int = Field(default=1, ge=1)
    max_concurrency: int = Field(default=4, ge=1)
    task_max_attempts: int = Field(default=3, ge=1)

    @field_validator("dataset")
    @classmethod
    def validate_dataset_name(cls, value: str) -> str:
        if not value or value in {".", ".."} or Path(value).name != value:
            raise ValueError(
                "dataset must be a folder name from src/validation/benchmark"
            )
        return value


def load_run_config(config_name: str) -> tuple[str, BenchmarkRunConfig]:
    name = config_name.removesuffix(".yaml").removesuffix(".yml")
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(
            "config must be a file name from src/validation/run_configs"
        )

    path = RUN_CONFIGS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Run config {config_name!r} does not exist at {path}")

    with path.open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)

    if not isinstance(values, dict):
        raise ValueError(f"Run config {path} must contain a YAML mapping")

    config = BenchmarkRunConfig.model_validate(values)
    dataset_path = BENCHMARKS_DIR / config.dataset / "dataset.yaml"
    if not dataset_path.is_file():
        raise ValueError(f"Dataset {config.dataset!r} does not exist at {dataset_path}")

    return path.stem, config
