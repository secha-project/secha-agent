from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def extract_inputs(path: Path) -> list[Any]:
    with path.open(encoding="utf-8") as dataset_file:
        dataset = yaml.safe_load(dataset_file)

    if not isinstance(dataset, dict) or not isinstance(dataset.get("cases"), list):
        raise TypeError(f"{path} must contain a top-level 'cases' list")

    inputs = []
    for index, case in enumerate(dataset["cases"], start=1):
        if not isinstance(case, dict) or "inputs" not in case:
            raise ValueError(f"Case {index} in {path} must contain 'inputs'")
        inputs.append(case["inputs"])
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract case inputs from a benchmark dataset YAML file."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    result = yaml.safe_dump(
        extract_inputs(args.dataset), allow_unicode=True, sort_keys=False
    )
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
