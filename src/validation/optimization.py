"""GEPA integration using the ordinary SECHA benchmark task and evaluators."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from gepa.core.adapter import EvaluationBatch, ProposalFn
from pydantic import TypeAdapter
from pydantic_ai import Agent, ModelSettings, UsageLimits
from pydantic_ai.models import Model
from pydantic_evals.reporting import ReportCase

from src.agent.agent import create_agent
from src.agent.deps import Deps
from src.agent.prompt import prompt_files
from src.validation.config import BenchmarkRunConfig
from src.validation.eval_models import BenchmarkCase, BenchmarkDataset
from src.validation.reporting import BenchmarkReport, save_report
from src.validation.runner import AgentTask, evaluate

Trajectory = dict[str, Any]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def split_cases(
    dataset: BenchmarkDataset,
    train: Sequence[str],
    validation: Sequence[str],
    test: Sequence[str] = (),
) -> tuple[list[BenchmarkCase], list[BenchmarkCase], list[BenchmarkCase]]:
    """Require explicit, non-overlapping case splits, retaining their order."""
    by_name = {case.name: case for case in dataset.cases}
    if None in by_name or len(by_name) != len(dataset.cases):
        raise ValueError("Optimization cases must have unique names")
    if not train or not validation:
        raise ValueError("Nonempty train and validation splits are required")
    names = [*train, *validation, *test]
    if len(names) != len(set(names)):
        raise ValueError(
            "Train, validation, and test cases must be disjoint without duplicates"
        )
    unknown = set(names) - by_name.keys()
    if unknown:
        raise ValueError(f"Unknown benchmark cases: {sorted(unknown)}")
    # Repetitions/aliases of the same source case must not cross split boundaries.
    selected_metadata = [by_name[name].metadata for name in names]
    if any(metadata is None for metadata in selected_metadata):
        raise ValueError("Optimization cases require benchmark metadata")
    ids = [
        str(metadata.case_id) for metadata in selected_metadata if metadata is not None
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("Selected cases must have distinct case IDs")
    return (
        [by_name[name] for name in train],
        [by_name[name] for name in validation],
        [by_name[name] for name in test],
    )


def report_batch(
    report: BenchmarkReport,
    batch: list[BenchmarkCase],
    *,
    capture_traces: bool,
    repetitions: int = 1,
) -> EvaluationBatch[Trajectory, str]:
    """Convert reports to GEPA's input order, including failed executions."""
    if repetitions > 1:
        outputs, scores, trajectories = [], [], []
        for case in batch:
            attempts = [
                BenchmarkCase(
                    name=f"{case.name} [{index}/{repetitions}]",
                    inputs=case.inputs,
                    expected_output=case.expected_output,
                    metadata=case.metadata,
                )
                for index in range(1, repetitions + 1)
            ]
            result = report_batch(report, attempts, capture_traces=capture_traces)
            outputs.append(json.dumps(result.outputs, ensure_ascii=False))
            scores.append(sum(result.scores) / repetitions)
            if capture_traces:
                trajectories.append(
                    {
                        "case_name": case.name,
                        "input": case.inputs,
                        "score": scores[-1],
                        "attempts": result.trajectories,
                    }
                )
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )
    results = {case.name: case for case in [*report.cases, *report.failures]}
    outputs: list[str] = []
    scores: list[float] = []
    trajectories: list[Trajectory] = []
    adapter = TypeAdapter(BenchmarkReport)
    serialized = adapter.dump_python(report, mode="json")
    records = {
        case["name"]: case for case in [*serialized["cases"], *serialized["failures"]]
    }
    for case in batch:
        if case.name is None:
            raise ValueError("Optimization cases require names")
        result = results[case.name]
        succeeded = isinstance(result, ReportCase)
        passed = (
            succeeded
            and not result.evaluator_failures
            and bool(result.assertions)
            and all(assertion.value for assertion in result.assertions.values())
        )
        outputs.append(str(result.output) if succeeded else "")
        scores.append(float(passed))
        if capture_traces:
            record = records[case.name]
            trajectories.append(
                {
                    "case_name": case.name,
                    "input": case.inputs,
                    "expected_output": record.get("expected_output"),
                    "reference_sql": case.metadata.query_code
                    if case.metadata
                    else None,
                    "actual_output": outputs[-1],
                    "score": scores[-1],
                    "assertions": record.get("assertions", {}),
                    "labels": record.get("labels", {}),
                    "evaluator_failures": record.get("evaluator_failures", []),
                    "error": record.get("error_message"),
                    "attributes": record.get("attributes", {}),
                }
            )
    return EvaluationBatch(
        outputs=outputs,
        scores=scores,
        trajectories=trajectories if capture_traces else None,
    )


@dataclass
class BenchmarkAdapter:
    config: BenchmarkRunConfig
    dataset: BenchmarkDataset
    baseline: dict[str, str]
    deps: Deps
    model: Model
    proposer: Agent[None, str]
    run_directory: Path
    reflection_character_limit: int = 30_000
    propose_new_texts: ProposalFn | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.propose_new_texts = self._propose_new_texts
        self.baseline = dict(self.baseline)
        if (
            self.baseline.keys()
            != prompt_files(self.config.agent.enabled_capabilities).keys()
        ):
            raise ValueError("The baseline must snapshot every active prompt component")
        (self.run_directory / "evaluations").mkdir(exist_ok=True)
        (self.run_directory / "proposals").mkdir(exist_ok=True)

    def evaluate(
        self,
        batch: list[BenchmarkCase],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, str]:
        prompts = self.baseline | candidate
        agent = create_agent(
            self.config.agent.enabled_capabilities, prompt_overrides=prompts
        )
        directory = Path(
            mkdtemp(prefix="batch-", dir=self.run_directory / "evaluations")
        )
        (directory / "artifacts").mkdir()
        write_json(directory / "prompts.json", prompts)
        dataset = BenchmarkDataset(
            name=self.dataset.name,
            cases=batch,
            evaluators=self.dataset.evaluators,
            report_evaluators=self.dataset.report_evaluators,
        )
        task = AgentTask(
            agent=agent,
            deps=self.deps,
            model=self.model,
            model_settings=ModelSettings(temperature=self.config.agent.temperature),
            usage_limits=UsageLimits(request_limit=self.config.agent.request_limit),
            capture_messages=capture_traces,
        )
        report = evaluate(
            dataset,
            task,
            directory.name,
            self.config,
            directory / "artifacts",
            directory / "checkpoint.json",
        )
        save_report(report, directory / "report.json")
        result = report_batch(
            report,
            batch,
            capture_traces=capture_traces,
            repetitions=self.config.repetitions,
        )
        write_json(
            directory / "scores.json",
            dict(
                zip(
                    [case.name for case in batch],
                    result.scores,
                    strict=True,
                )
            ),
        )
        return result

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, str],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if eval_batch.trajectories is None:
            raise ValueError("Reflection requires captured trajectories")
        records = []
        for case_trajectory in eval_batch.trajectories:
            attempts = case_trajectory.get("attempts", [case_trajectory])
            for trajectory in attempts:
                records.append(self._reflection_record(trajectory))
        return {component: records for component in components_to_update}

    def _reflection_record(self, trajectory: Trajectory) -> dict[str, Any]:
        feedback = {
            key: value for key, value in trajectory.items() if key != "attributes"
        }
        attributes = trajectory["attributes"]
        # Keep evaluator feedback ahead of potentially long SQL results/tool traces.
        return {
            "feedback": feedback,
            "execution": json.dumps(attributes, ensure_ascii=False)[
                : self.reflection_character_limit
            ],
            "execution_may_be_truncated": len(
                json.dumps(attributes, ensure_ascii=False)
            )
            > self.reflection_character_limit,
        }

    def _propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        replacements = {}
        for component in components_to_update:
            request = {
                "component": component,
                "enabled_capabilities": sorted(self.config.agent.enabled_capabilities),
                "current_prompts": self.baseline | candidate,
                "feedback": reflective_dataset[component],
            }
            result = self.proposer.run_sync(
                json.dumps(request, ensure_ascii=False),
                usage_limits=UsageLimits(request_limit=1),
            )
            text = result.output.strip()
            error = None
            try:
                if text.startswith("```"):
                    raise ValueError(
                        "Return plain prompt text without Markdown code fences"
                    )
                create_agent(
                    self.config.agent.enabled_capabilities,
                    prompt_overrides=self.baseline | candidate | {component: text},
                )
            except ValueError as exc:
                error = str(exc)
                text = candidate[component]
            replacements[component] = text
            directory = Path(
                mkdtemp(prefix="proposal-", dir=self.run_directory / "proposals")
            )
            write_json(
                directory / "proposal.json",
                {
                    "component": component,
                    "proposed_text": result.output,
                    "validation_error": error,
                    "usage": asdict(result.usage),
                },
            )
        return replacements
