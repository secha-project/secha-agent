"""Optimize prompts with frozen inputs, bounded retries, and resumable evaluations."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import shlex
import shutil
import time
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from gepa import optimize
from gepa.core.result import GEPAResult
from gepa.strategies.candidate_selector import ParetoCandidateSelector
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.exceptions import ModelAPIError

from src.agent.agent import create_agent
from src.agent.capabilities import CAPABILITIES
from src.agent.prompt import prompt_files, render_prompt, snapshot_prompts
from src.validation.config import BENCHMARKS_DIR, PROJECT_ROOT, load_run_config
from src.validation.eval_models import BenchmarkDataset
from src.validation.evaluators import ValidateVisualizationQuality
from src.validation.optimization import (
    BenchmarkAdapter,
    create_component_selector,
    split_cases,
    write_json,
)
from src.validation.optimization_state import (
    OptimizationProgressLog,
    OptimizationServiceError,
    lock_run,
    retryable_service_error,
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
FORMAT_VERSION = 4
COMPONENT_SELECTION = "feedback"
SNAPSHOT_FILES = (
    "baseline.json",
    "seed-candidate.json",
    "proposer-instructions.md",
    "selector-instructions.md",
)
DEFAULTS = {
    "max_metric_calls": 100,
    "reflection_minibatch_size": 3,
    "seed": 0,
    "model_request_max_attempts": 20,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Target benchmark run config name")
    parser.add_argument(
        "--reflection-model",
        help="Model name for component selection and rewriting, e.g. gpt-6-sol",
    )
    parser.add_argument(
        "--reflection-provider",
        choices=("openai", "zai", "tuni"),
        help="Optional provider override; inferred from GPT/o-series, GLM, or Qwen model names",
    )
    location = parser.add_mutually_exclusive_group()
    location.add_argument(
        "--resume", type=Path, help="Continue a saved optimization directory"
    )
    location.add_argument(
        "--results-dir",
        type=Path,
        help="Root for new runs, including all logs, evaluation artifacts, and checkpoints (default: project benchmark-results)",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        help="YAML mapping containing train, validation, and test case lists",
    )
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}", nargs="+", metavar="CASE")
    parser.add_argument(
        "--components", nargs="+", help="Named components to optimize (default: all)"
    )
    parser.add_argument("--max-metric-calls", type=int)
    parser.add_argument("--reflection-minibatch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--model-request-max-attempts",
        type=int,
        help="Optimizer-only request retry ceiling (default: 20)",
    )
    parser.add_argument(
        "--max-resumes",
        type=int,
        default=3,
        help="Automatic restarts after service failures (default: 3; 0 disables)",
    )
    parser.add_argument("--resume-delay-seconds", type=float, default=60)
    parser.add_argument(
        "--progress-log", type=Path, help="Summary log (default: RUN/progress.log)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/save inputs without external calls",
    )
    parser.add_argument("--list-components", action="store_true")
    args = parser.parse_args(argv)
    if args.list_components:
        return args
    if args.progress_log is not None:
        args.progress_log = args.progress_log.expanduser().resolve()
    if (
        args.max_resumes < 0
        or not math.isfinite(args.resume_delay_seconds)
        or args.resume_delay_seconds < 0
    ):
        parser.error("Resume count and delay must be nonnegative")
    if args.resume:
        if any(
            getattr(args, key) is not None
            for key in (
                *DEFAULTS,
                "train",
                "validation",
                "test",
                "components",
                "split_file",
            )
        ):
            parser.error(
                "--resume uses saved splits, components, seed, budgets, and retry limits"
            )
        return args
    if not args.config:
        parser.error("--config is required")
    if not args.dry_run and not args.reflection_model:
        parser.error("--reflection-model is required unless --dry-run is used")
    if args.reflection_provider and not args.reflection_model:
        parser.error("--reflection-provider requires --reflection-model")
    if args.split_file:
        if any(getattr(args, split) for split in ("train", "validation", "test")):
            parser.error("--split-file cannot be combined with explicit case lists")
        splits = yaml.safe_load(args.split_file.read_text())
        if not isinstance(splits, dict):
            parser.error("Split file must be a mapping")
        for split in ("train", "validation", "test"):
            values = splits.get(split, [])
            if not isinstance(values, list) or not all(
                isinstance(value, str) for value in values
            ):
                parser.error(f"{split} must be a list of case names")
            setattr(args, split, values)
        args.split_dataset = splits.get("dataset")
        for group in splits.get("groups", []):
            if (
                not isinstance(group, list)
                or not group
                or not all(isinstance(name, str) for name in group)
            ):
                parser.error("Split groups must be nonempty lists of case names")
            if (
                sum(
                    set(group) <= set(getattr(args, split))
                    for split in ("train", "validation", "test")
                )
                != 1
            ):
                parser.error("Every declared group must stay together in one split")
    for split in ("train", "validation", "test"):
        setattr(args, split, getattr(args, split) or [])
    for key, default in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, default)
    if not args.train or not args.validation:
        parser.error("--train and --validation are required (or provide --split-file)")
    if args.max_metric_calls < len(args.validation):
        parser.error("--max-metric-calls must cover the initial validation set")
    if min(args.reflection_minibatch_size, args.model_request_max_attempts) <= 0:
        parser.error("Minibatch size and model request limits must be positive")
    return args


def file_hashes(paths: list[Path], root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def source_hashes() -> dict[str, str]:
    return file_hashes(
        [
            *sorted((PROJECT_ROOT / "src").rglob("*.py")),
            PROJECT_ROOT / "pyproject.toml",
            PROJECT_ROOT / "uv.lock",
        ],
        PROJECT_ROOT,
    )


def dataset_hashes(directory: Path) -> dict[str, str]:
    return file_hashes(
        sorted(path for path in directory.rglob("*") if path.is_file()), directory
    )


def dependency_versions() -> dict[str, str]:
    return {
        name: version(name)
        for name in (
            "gepa",
            "pydantic",
            "pydantic-ai-slim",
            "pydantic-evals",
            "ggsql",
            "pyspark-client",
        )
    }


def bounded_agent(agent, args):
    return agent.model_copy(
        update={
            "model_request_max_attempts": args.model_request_max_attempts,
        }
    )


def reflection_provider(model_name: str, explicit: str | None = None) -> str:
    if not model_name.strip() or model_name != model_name.strip():
        raise ValueError(
            "Reflection model name must be nonempty without surrounding whitespace"
        )
    if explicit is not None:
        return explicit
    name = model_name.lower()
    if name.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
        return "openai"
    if name.startswith("glm-"):
        return "zai"
    if name.startswith("qwen"):
        return "tuni"
    raise ValueError(
        f"Cannot infer a provider for reflection model {model_name!r}; use --reflection-provider"
    )


def reflection_agent(target, args):
    """Reuse provider construction and retry handling without a benchmark config."""
    if args.reflection_model is None:
        return None
    args.reflection_provider = reflection_provider(
        args.reflection_model, args.reflection_provider
    )
    return target.model_copy(
        update={
            "model_provider": args.reflection_provider,
            "model_name": args.reflection_model,
            "temperature": None,
            "pricing": None,
            "enabled_capabilities": frozenset(),
        }
    )


def runtime_endpoints(config) -> dict:
    # No credentials or URL query parameters belong in the manifest.
    result = {}
    for name in ("catalog_url", "spark_url"):
        url = urlsplit(getattr(config.agent, name))
        result[name] = [url.scheme, url.hostname, url.port]
    return result


def prepare_run(args):
    saved = None
    if args.resume:
        directory = args.resume.expanduser().resolve()
        saved = json.loads((directory / "manifest.json").read_text())
        if saved.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                "This directory uses an incompatible optimization format; start a new run"
            )
        for key in ("config", "reflection_model", "reflection_provider"):
            if getattr(args, key) is not None and getattr(args, key) != saved[key]:
                raise ValueError(f"Resume {key} differs from the saved run")
            setattr(args, key, saved[key])
        for key in DEFAULTS:
            setattr(args, key, saved[key])
        for split, cases in saved["splits"].items():
            setattr(args, split, cases)
        args.components = saved["components"]
        if args.progress_log is None:
            args.progress_log = Path(saved["progress_log"])
        if not args.reflection_model and not args.dry_run:
            raise ValueError(
                "This dry run has no reflection model; create a new run with --reflection-model"
            )

    name, original = load_run_config(args.config)
    if getattr(args, "split_dataset", None) not in (None, original.dataset):
        raise ValueError("Split file belongs to a different dataset")
    config = original.model_copy(update={"agent": bounded_agent(original.agent, args)})
    reflection = reflection_agent(config.agent, args)
    selected = config.agent.enabled_capabilities
    if saved:
        checks = {
            "component_selection": COMPONENT_SELECTION,
            "dependency_versions": dependency_versions(),
            "source_sha256": source_hashes(),
            "dataset_sha256": dataset_hashes(directory / "dataset"),
            "target": config.agent.experiment_metadata(),
            "reflection": reflection.experiment_metadata() if reflection else None,
            "benchmark": config.model_dump(mode="json", exclude={"agent"}),
            "runtime_endpoints": runtime_endpoints(config),
            "snapshot_sha256": file_hashes(
                [directory / name for name in SNAPSHOT_FILES],
                directory,
            ),
        }
        for key, value in checks.items():
            if saved[key] != value:
                raise ValueError(
                    f"Cannot resume: {key} changed since the run was created"
                )
        baseline = json.loads((directory / "baseline.json").read_text())
        proposer_instructions = (
            (directory / "proposer-instructions.md").read_text().rstrip()
        )
        selector_instructions = (
            (directory / "selector-instructions.md").read_text().rstrip()
        )
    else:
        baseline = snapshot_prompts(selected)
        args.components = (
            args.components if args.components is not None else list(baseline)
        )
        proposer_instructions = render_prompt("optimization/proposer.md", selected)
        selector_instructions = render_prompt("optimization/selector.md", selected)
    if not args.components or len(args.components) != len(set(args.components)):
        raise ValueError("Components must be nonempty and unique")
    if set(args.components) - baseline.keys():
        raise ValueError("Unknown or inactive prompt components")
    create_agent(selected, prompt_overrides=baseline)
    source_dataset = load_dataset(config.dataset)
    if not saved:
        split_cases(source_dataset, args.train, args.validation, args.test)
        directory = create_run_directory(
            f"gepa-{name}", config.dataset, results_dir=args.results_dir
        )
        shutil.copytree(BENCHMARKS_DIR / config.dataset, directory / "dataset")
        write_json(directory / "baseline.json", baseline)
        write_json(
            directory / "seed-candidate.json",
            {key: baseline[key] for key in args.components},
        )
        (directory / "proposer-instructions.md").write_text(
            proposer_instructions + "\n", encoding="utf-8"
        )
        (directory / "selector-instructions.md").write_text(
            selector_instructions + "\n", encoding="utf-8"
        )
        write_json(
            directory / "manifest.json",
            {
                "format_version": FORMAT_VERSION,
                "component_selection": COMPONENT_SELECTION,
                "dependency_versions": dependency_versions(),
                "progress_log": str(
                    (args.progress_log or directory / "progress.log").resolve()
                ),
                "config": args.config,
                "reflection_model": args.reflection_model,
                "reflection_provider": args.reflection_provider,
                "target": config.agent.experiment_metadata(),
                "reflection": reflection.experiment_metadata() if reflection else None,
                "benchmark": config.model_dump(mode="json", exclude={"agent"}),
                "runtime_endpoints": runtime_endpoints(config),
                "splits": {
                    split: getattr(args, split)
                    for split in ("train", "validation", "test")
                },
                "components": args.components,
                "prompt_files": prompt_files(selected),
                **{key: getattr(args, key) for key in DEFAULTS},
                "source_sha256": source_hashes(),
                "dataset_sha256": dataset_hashes(directory / "dataset"),
                "snapshot_sha256": file_hashes(
                    [directory / name for name in SNAPSHOT_FILES],
                    directory,
                ),
            },
        )
    dataset = BenchmarkDataset.from_file(
        directory / "dataset" / "dataset.yaml",
        custom_evaluator_types=CUSTOM_EVALUATOR_TYPES,
    )
    attach_case_evaluators(dataset, directory / "dataset")
    dataset.report_evaluators = source_dataset.report_evaluators
    for case in dataset.cases:
        for evaluator in case.evaluators:
            if isinstance(evaluator, ValidateVisualizationQuality):
                evaluator.model_request_max_attempts = args.model_request_max_attempts
    splits = split_cases(dataset, args.train, args.validation, args.test)
    return (
        directory,
        config,
        reflection,
        baseline,
        proposer_instructions,
        selector_instructions,
        dataset,
        splits,
    )


def export_winner(directory, selected, baseline, result):
    best_prompts = baseline | result.candidates[result.best_idx]
    write_json(directory / "best-prompts.json", best_prompts)
    changes = []
    for key, relative in prompt_files(selected).items():
        export = directory / "prompts" / relative
        export.parent.mkdir(parents=True, exist_ok=True)
        export.write_text(best_prompts[key].rstrip() + "\n", encoding="utf-8")
        changes.extend(
            difflib.unified_diff(
                (baseline[key].rstrip() + "\n").splitlines(keepends=True),
                (best_prompts[key].rstrip() + "\n").splitlines(keepends=True),
                fromfile=f"baseline/{relative}",
                tofile=f"best/{relative}",
            )
        )
    (directory / "prompt-changes.diff").write_text("".join(changes), encoding="utf-8")


def execute_search(args, prepared, progress):
    (
        directory,
        config,
        reflection,
        baseline,
        instructions,
        selector_instructions,
        dataset,
        splits,
    ) = prepared
    train, validation, test = splits
    result_path = directory / "result.json"
    result = (
        GEPAResult.from_dict(json.loads(result_path.read_text()))
        if result_path.exists()
        else None
    )
    heldout_path = directory / "held-out-comparison.json"
    if result is None or (test and not heldout_path.exists()):
        assert reflection is not None
        reflection_model = reflection.create_model()
        reflection_settings = (
            ModelSettings(temperature=reflection.temperature)
            if reflection.temperature is not None
            else ModelSettings()
        )
        proposer = Agent(
            reflection_model,
            output_type=str,
            instructions=instructions,
            model_settings=reflection_settings,
        )
        adapter = BenchmarkAdapter(
            config=config,
            dataset=dataset,
            baseline=baseline,
            deps=create_dependencies(config.agent),
            model=config.agent.create_model(),
            proposer=proposer,
            selector=create_component_selector(
                reflection_model, selector_instructions, reflection_settings
            ),
            run_directory=directory,
            seed=args.seed,
            reflection_minibatch_size=min(args.reflection_minibatch_size, len(train)),
            progress=progress,
        )
        seed_candidate = {key: baseline[key] for key in args.components}
        if result is None:
            result = optimize(
                seed_candidate=seed_candidate,
                trainset=train,
                valset=validation,
                adapter=adapter,
                max_metric_calls=args.max_metric_calls,
                candidate_selection_strategy=ParetoCandidateSelector(adapter.rng),
                batch_sampler=adapter.sampler,
                module_selector=adapter.select_components,
                seed=args.seed,
                run_dir=str(directory / "gepa"),
                raise_on_exception=True,
                use_merge=False,
                cache_evaluation=False,
                callbacks=[progress],
                stop_callbacks=[adapter.check_service_health],
            )
            write_json(result_path, result.to_dict())
        export_winner(directory, config.agent.enabled_capabilities, baseline, result)
        if test and not heldout_path.exists():
            adapter.set_evaluation_phase("held-out-baseline")
            original = adapter.evaluate(test, seed_candidate)
            adapter.set_evaluation_phase("held-out-best")
            optimized = adapter.evaluate(test, result.candidates[result.best_idx])
            write_json(
                heldout_path,
                {
                    "cases": args.test,
                    "baseline_scores": original.scores,
                    "best_scores": optimized.scores,
                    "repetitions": config.repetitions,
                    "note": "These evaluations are outside the GEPA search budget.",
                },
            )
    else:
        export_winner(directory, config.agent.enabled_capabilities, baseline, result)
    progress.update(
        phase="complete",
        batch=None,
        metric_calls=result.total_metric_calls,
        best_validation=result.val_aggregate_scores[result.best_idx],
    )
    print(
        f"Validation score: {result.val_aggregate_scores[0]:.3f} -> {result.val_aggregate_scores[result.best_idx]:.3f}"
    )
    print(
        f"Saved best prompts and review diff to {directory}. Project prompt files were not changed."
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_components:
        selected = (
            load_run_config(args.config)[1].agent.enabled_capabilities
            if args.config
            else ALL_CAPABILITIES
        )
        print("\n".join(prompt_files(selected)))
        return
    prepared = prepare_run(args)
    directory, config = prepared[:2]
    print(f"Optimization directory: {directory}", flush=True)
    print(
        f"Capabilities: {', '.join(sorted(config.agent.enabled_capabilities))}; {len(args.components)} components; {config.repetitions} attempt(s) per case."
    )
    reflection = prepared[2]
    task_model = config.agent.model_name or config.agent.load_provider().model_name
    reflection_model = (
        reflection.model_name or reflection.load_provider().model_name
        if reflection is not None
        else None
    )
    print(f"Task model: {task_model} (--config {args.config})")
    if reflection is not None:
        print(
            f"Selection and rewrite model: {reflection_model} (provider: {args.reflection_provider})"
        )
    if args.dry_run:
        print("Dry run complete. No model or database calls were made.")
        return
    log_path = args.progress_log or directory / "progress.log"
    print(f"Progress log: {log_path}", flush=True)
    with (
        lock_run(directory),
        OptimizationProgressLog(
            log_path, directory.name, directory, args.max_metric_calls
        ) as progress,
    ):
        progress.update(
            task_model=task_model,
            reflection_model=reflection_model,
            component_selection=COMPONENT_SELECTION,
        )
        progress.event(
            "MODELS", "reflection model selects components and rewrites prompts"
        )
        configure_observability(config.agent)
        try:
            for restart in range(args.max_resumes + 1):
                try:
                    execute_search(args, prepared, progress)
                    break
                except (OptimizationServiceError, ModelAPIError) as exc:
                    retryable = (
                        exc.retryable
                        if isinstance(exc, OptimizationServiceError)
                        else retryable_service_error(str(exc))
                    )
                    if not retryable or restart >= args.max_resumes:
                        raise
                    progress.update(
                        phase="waiting-to-resume", automatic_resumes=restart + 1
                    )
                    progress.event(
                        "PAUSED",
                        f"{type(exc).__name__}; retrying in {args.resume_delay_seconds:g}s",
                    )
                    time.sleep(args.resume_delay_seconds)
                    progress.event("RESUMING")
        except BaseException:
            print(
                f"Continue with: python -m src.validation.run_optimization --resume {shlex.quote(str(directory))}",
                flush=True,
            )
            raise


if __name__ == "__main__":
    main()
