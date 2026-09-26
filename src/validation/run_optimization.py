"""Run offline prompt optimization for a selected benchmark/model configuration."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import shutil

from gepa import optimize
from pydantic_ai import Agent, ModelSettings

from src.agent.agent import create_agent
from src.agent.capabilities import CAPABILITIES
from src.agent.prompt import prompt_files, render_prompt, snapshot_prompts
from src.validation.config import BENCHMARKS_DIR, PROJECT_ROOT, load_run_config
from src.validation.eval_models import BenchmarkDataset
from src.validation.optimization import (
    BenchmarkAdapter,
    split_cases,
    write_json,
)
from src.validation.runner import (
    CUSTOM_EVALUATOR_TYPES,
    attach_case_evaluators,
    configure_observability,
    create_dependencies,
    create_run_directory,
    load_dataset,
)

ALL_CAPABILITIES = frozenset(capability.NAME for capability in CAPABILITIES)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", help="Target model's existing benchmark run config name"
    )
    parser.add_argument(
        "--reflection-config",
        help="Benchmark run config whose agent/model proposes edits",
    )
    parser.add_argument("--train", nargs="+", default=[], metavar="CASE")
    parser.add_argument("--validation", nargs="+", default=[], metavar="CASE")
    parser.add_argument("--test", nargs="+", default=[], metavar="CASE")
    parser.add_argument(
        "--components", nargs="+", help="Named components to optimize (default: all)"
    )
    parser.add_argument("--max-metric-calls", type=int, default=100)
    parser.add_argument("--reflection-minibatch-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Save and validate inputs without model or database calls",
    )
    parser.add_argument(
        "--list-components",
        action="store_true",
        help="List prompt names and exit; no configuration needed",
    )
    args = parser.parse_args(argv)
    if args.list_components:
        return args
    if not args.config:
        parser.error("--config is required")
    if not args.dry_run and not args.reflection_config:
        parser.error("--reflection-config is required unless --dry-run is used")
    if not args.train or not args.validation:
        parser.error("--train and --validation are required")
    if args.max_metric_calls < len(args.validation):
        parser.error(
            "--max-metric-calls must cover at least the initial validation set"
        )
    if args.reflection_minibatch_size < 1:
        parser.error("--reflection-minibatch-size must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_components:
        selected = (
            load_run_config(args.config)[1].agent.enabled_capabilities
            if args.config
            else ALL_CAPABILITIES
        )
        for key in prompt_files(selected):
            print(key)
        return

    name, config = load_run_config(args.config)
    selected = config.agent.enabled_capabilities
    reflection = (
        load_run_config(args.reflection_config)[1].agent
        if args.reflection_config
        else None
    )
    baseline = snapshot_prompts(selected)
    components = args.components if args.components is not None else list(baseline)
    if len(components) != len(set(components)) or not components:
        raise ValueError("Components must be nonempty and unique")
    unknown = set(components) - baseline.keys()
    if unknown:
        raise ValueError(f"Unknown prompt components: {sorted(unknown)}")
    create_agent(selected, prompt_overrides=baseline)
    source_dataset = load_dataset(config.dataset)
    split_cases(source_dataset, args.train, args.validation, args.test)

    directory = create_run_directory(f"gepa-{name}", config.dataset)
    shutil.copytree(BENCHMARKS_DIR / config.dataset, directory / "dataset")
    dataset = BenchmarkDataset.from_file(
        directory / "dataset" / "dataset.yaml",
        custom_evaluator_types=CUSTOM_EVALUATOR_TYPES,
    )
    attach_case_evaluators(dataset, directory / "dataset")
    dataset.report_evaluators = source_dataset.report_evaluators
    train, validation, test = split_cases(
        dataset, args.train, args.validation, args.test
    )
    seed_candidate = {key: baseline[key] for key in components}
    write_json(directory / "baseline.json", baseline)
    write_json(directory / "seed-candidate.json", seed_candidate)
    proposer_instructions = render_prompt("optimization/proposer.md", selected)
    (directory / "proposer-instructions.md").write_text(
        proposer_instructions + "\n", encoding="utf-8"
    )
    source_files = [
        *sorted((PROJECT_ROOT / "src").rglob("*.py")),
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "uv.lock",
    ]
    write_json(
        directory / "manifest.json",
        {
            "target": config.agent.experiment_metadata(),
            "reflection": reflection.experiment_metadata() if reflection else None,
            "benchmark": config.model_dump(mode="json", exclude={"agent"}),
            "splits": {
                "train": args.train,
                "validation": args.validation,
                "test": args.test,
            },
            "components": components,
            "prompt_files": prompt_files(selected),
            "max_metric_calls": args.max_metric_calls,
            "reflection_minibatch_size": args.reflection_minibatch_size,
            "seed": args.seed,
            "source_sha256": {
                str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in source_files
            },
            "dataset_sha256": {
                str(path.relative_to(directory / "dataset")): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted((directory / "dataset").rglob("*"))
                if path.is_file()
            },
        },
    )
    print(f"Optimization directory: {directory}")
    print(
        f"Capabilities: {', '.join(sorted(selected))}; "
        f"{len(components)} components; {config.repetitions} attempt(s) per case."
    )
    if args.dry_run:
        print("Dry run complete. No model or database calls were made.")
        return

    assert reflection is not None
    configure_observability(config.agent)
    proposer = Agent(
        reflection.create_model(),
        output_type=str,
        instructions=proposer_instructions,
        model_settings=(
            ModelSettings(temperature=reflection.temperature)
            if reflection.temperature is not None
            else ModelSettings()
        ),
    )
    adapter = BenchmarkAdapter(
        config=config,
        dataset=dataset,
        baseline=baseline,
        deps=create_dependencies(config.agent),
        model=config.agent.create_model(),
        proposer=proposer,
        run_directory=directory,
    )
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=train,
        valset=validation,
        adapter=adapter,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=min(args.reflection_minibatch_size, len(train)),
        module_selector="round_robin",
        seed=args.seed,
        run_dir=str(directory / "gepa"),
        raise_on_exception=True,
        use_merge=False,
        cache_evaluation=False,
    )
    best = result.candidates[result.best_idx]
    best_prompts = baseline | best
    write_json(directory / "result.json", result.to_dict())
    write_json(directory / "best-prompts.json", best_prompts)
    changes = []
    for key, relative_path in prompt_files(selected).items():
        export = directory / "prompts" / relative_path
        export.parent.mkdir(parents=True, exist_ok=True)
        export.write_text(best_prompts[key].rstrip() + "\n", encoding="utf-8")
        changes.extend(
            difflib.unified_diff(
                (baseline[key].rstrip() + "\n").splitlines(keepends=True),
                (best_prompts[key].rstrip() + "\n").splitlines(keepends=True),
                fromfile=f"baseline/{relative_path}",
                tofile=f"best/{relative_path}",
            )
        )
    (directory / "prompt-changes.diff").write_text("".join(changes), encoding="utf-8")
    if test:
        original = adapter.evaluate(test, seed_candidate)
        optimized = adapter.evaluate(test, best)
        write_json(
            directory / "held-out-comparison.json",
            {
                "cases": args.test,
                "baseline_scores": original.scores,
                "best_scores": optimized.scores,
                "repetitions": config.repetitions,
                "note": "These evaluations are outside the GEPA search budget.",
            },
        )
    print(
        f"Validation score: {result.val_aggregate_scores[0]:.3f} -> {result.val_aggregate_scores[result.best_idx]:.3f}"
    )
    print(
        f"Saved best prompts and review diff to {directory}. Project prompt files were not changed."
    )


if __name__ == "__main__":
    main()
