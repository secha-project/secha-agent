"""GEPA integration using the ordinary SECHA benchmark task and evaluators."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gepa.core.adapter import EvaluationBatch, ProposalFn
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic_ai import (
    Agent,
    ModelRetry,
    ModelSettings,
    PromptedOutput,
    RunContext,
    UsageLimits,
)
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models import Model
from pydantic_evals.reporting import ReportCase

from src.agent.agent import create_agent
from src.agent.deps import Deps
from src.agent.prompt import prompt_files
from src.validation.config import BenchmarkRunConfig
from src.validation.eval_models import BenchmarkCase, BenchmarkDataset
from src.validation.optimization_state import (
    OptimizationProgressLog,
    OptimizationServiceError,
    clear_service_failures,
    operation_directory,
    retryable_service_error,
    write_json,
)
from src.validation.reporting import BenchmarkReport, save_report
from src.validation.runner import AgentTask, evaluate

Trajectory = dict[str, Any]


class ComponentSelection(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    component: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


def create_component_selector(
    model: Model, instructions: str, model_settings: ModelSettings
) -> Agent[tuple[str, ...], ComponentSelection]:
    # Prompted JSON works with the same text-generation interface as the proposer.
    selector = Agent(
        model,
        deps_type=tuple[str, ...],
        output_type=PromptedOutput(ComponentSelection),
        instructions=instructions,
        model_settings=model_settings,
        retries=2,
    )

    @selector.output_validator
    def validate_component(
        ctx: RunContext[tuple[str, ...]], selection: ComponentSelection
    ) -> ComponentSelection:
        if selection.component not in ctx.deps:
            raise ModelRetry(f"Choose exactly one editable component: {list(ctx.deps)}")
        return selection

    return selector


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
    artifact_base_path: Path,
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
            result = report_batch(
                report,
                attempts,
                artifact_base_path=artifact_base_path,
                capture_traces=capture_traces,
            )
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
                    "reference_sql": (
                        artifact_base_path / case.metadata.query_path
                    ).read_text(encoding="utf-8")
                    if case.metadata and case.metadata.query_path is not None
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
    selector: Agent[tuple[str, ...], ComponentSelection] | None = None
    reflection_character_limit: int = 30_000
    seed: int = 0
    reflection_minibatch_size: int = 3
    progress: OptimizationProgressLog | None = None
    propose_new_texts: ProposalFn | None = field(init=False, repr=False)
    rng: random.Random = field(init=False, repr=False)
    sampler: EpochShuffledBatchSampler = field(init=False, repr=False)
    _evaluation_index: int = field(init=False, default=0)
    _proposal_index: int = field(init=False, default=0)
    _selection_index: int = field(init=False, default=0)
    _selection: dict[str, Any] | None = field(init=False, default=None)
    evaluation_phase: str = field(init=False, default="search")
    _service_error: ModelAPIError | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.propose_new_texts = self._propose_new_texts
        self.rng = random.Random(self.seed)
        self.sampler = EpochShuffledBatchSampler(
            self.reflection_minibatch_size, self.rng
        )
        self.baseline = dict(self.baseline)
        if (
            self.baseline.keys()
            != prompt_files(self.config.agent.enabled_capabilities).keys()
        ):
            raise ValueError("The baseline must snapshot every active prompt component")
        (self.run_directory / "evaluations").mkdir(exist_ok=True)
        (self.run_directory / "proposals").mkdir(exist_ok=True)
        (self.run_directory / "selections").mkdir(exist_ok=True)

    def get_adapter_state(self) -> dict[str, Any]:
        return {
            "evaluation_index": self._evaluation_index,
            "proposal_index": self._proposal_index,
            "selection_index": self._selection_index,
            "rng": self.rng.getstate(),
            "sampler": deepcopy(
                {
                    key: value
                    for key, value in vars(self.sampler).items()
                    if key != "rng"
                }
            ),
        }

    def set_adapter_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        self._evaluation_index = state["evaluation_index"]
        self._proposal_index = state["proposal_index"]
        self._selection_index = state["selection_index"]
        self._selection = None
        self.rng.setstate(state["rng"])
        vars(self.sampler).update(deepcopy(state["sampler"]))

    def set_evaluation_phase(self, phase: str) -> None:
        self.evaluation_phase = phase
        self._evaluation_index = 0

    def check_service_health(self, gepa_state=None) -> bool:
        # GEPA 0.1.4 swallows reflection exceptions even with raise_on_exception.
        # Its first stop callback runs before the next state save, so raising here
        # retains the pre-iteration checkpoint and its random/sampler state.
        if self._service_error is not None:
            raise self._service_error
        return False

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
        directory = operation_directory(
            self.run_directory / "evaluations",
            f"{self.evaluation_phase}-{self._evaluation_index:06d}",
            {
                "prompts": prompts,
                "cases": [case.name for case in batch],
                "capture_traces": capture_traces,
            },
        )
        if self.progress:
            self.progress.update(
                phase=self.evaluation_phase,
                batch=str(directory),
                batch_total=len(batch) * self.config.repetitions,
            )
        saved_result = directory / "evaluation.json"
        if saved_result.exists():
            self._evaluation_index += 1
            return EvaluationBatch(**json.loads(saved_result.read_text()))
        (directory / "artifacts").mkdir(exist_ok=True)
        write_json(directory / "prompts.json", prompts)
        clear_service_failures(directory / "checkpoint.json")
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
            model_settings=(
                ModelSettings(temperature=self.config.agent.temperature)
                if self.config.agent.temperature is not None
                else ModelSettings()
            ),
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
        service_errors = clear_service_failures(directory / "checkpoint.json")
        if service_errors:
            raise OptimizationServiceError(
                f"Model/judge service failed in {directory.name}; {len(service_errors)} attempt(s) left pending. {service_errors[0]}",
                retryable=all(
                    retryable_service_error(message) for message in service_errors
                ),
            )
        result = report_batch(
            report,
            batch,
            artifact_base_path=self.run_directory / "dataset",
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
        write_json(saved_result, asdict(result))
        self._evaluation_index += 1
        return result

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, str],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if eval_batch.trajectories is None:
            raise ValueError("Reflection requires captured trajectories")
        records = self._reflection_records(eval_batch.trajectories)
        return {component: records for component in components_to_update}

    def _reflection_records(
        self, trajectories: list[Trajectory]
    ) -> list[dict[str, Any]]:
        records = []
        for case_trajectory in trajectories:
            attempts = case_trajectory.get("attempts", [case_trajectory])
            for trajectory in attempts:
                records.append(self._reflection_record(trajectory))
        return records

    def select_components(
        self,
        state,
        trajectories: list[Trajectory],
        subsample_scores: list[float],
        candidate_idx: int,
        candidate: dict[str, str],
    ) -> list[str]:
        """Choose one editable prompt using only the current training feedback."""
        self.check_service_health()
        self._selection = None
        if not candidate or candidate.keys() - self.baseline.keys():
            raise ValueError("Selection requires active, editable prompt components")
        request = {
            "candidate_index": candidate_idx,
            "editable_components": list(candidate),
            "enabled_capabilities": sorted(self.config.agent.enabled_capabilities),
            "current_prompts": self.baseline | candidate,
            "scores": subsample_scores,
            "feedback": self._reflection_records(trajectories),
        }
        operation = f"selection-{self._selection_index:06d}"
        directory = operation_directory(
            self.run_directory / "selections", operation, request
        )
        if self.progress:
            self.progress.update(
                phase="component-selection",
                component=None,
                batch=None,
                selections=self._selection_index,
            )
        saved = directory / "selection.json"
        if saved.exists():
            record = json.loads(saved.read_text())
            selection = ComponentSelection.model_validate(record["selection"])
        else:
            usage = None
            if len(candidate) == 1:
                selection = ComponentSelection(
                    component=next(iter(candidate)),
                    rationale="Only one component is editable in this experiment.",
                )
            else:
                if self.selector is None:
                    raise ValueError(
                        "Feedback-driven component selection requires a selector"
                    )
                # Selection exceptions propagate directly from GEPA's selector hook;
                # service failures therefore resume from the pre-iteration checkpoint.
                result = self.selector.run_sync(
                    json.dumps(request, ensure_ascii=False),
                    deps=tuple(candidate),
                    usage_limits=UsageLimits(request_limit=3),
                )
                selection = result.output
                usage = asdict(result.usage)
            record = {"selection": selection.model_dump(), "usage": usage}
            write_json(saved, record)
        if selection.component not in candidate:
            raise ValueError("Saved selection is not an editable component")
        self._selection = {"operation": operation, **selection.model_dump()}
        self._selection_index += 1
        if self.progress:
            self.progress.update(
                component=selection.component, selections=self._selection_index
            )
            rationale = " ".join(selection.rationale.split())[:500]
            self.progress.event(
                "SELECTION", f"{operation}: {selection.component}; {rationale}"
            )
        return [selection.component]

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
        self.check_service_health()
        replacements = {}
        for component in components_to_update:
            request = {
                "component": component,
                "enabled_capabilities": sorted(self.config.agent.enabled_capabilities),
                "current_prompts": self.baseline | candidate,
                "feedback": reflective_dataset[component],
            }
            if self._selection and self._selection["component"] == component:
                request["selection"] = self._selection
            directory = operation_directory(
                self.run_directory / "proposals",
                f"proposal-{self._proposal_index:06d}",
                request,
            )
            if self.progress:
                self.progress.update(
                    phase="reflection",
                    component=component,
                    proposals=self._proposal_index,
                    batch=None,
                )
                self.progress.event("PROPOSAL", component)
            saved = directory / "proposal.json"
            if saved.exists():
                replacements[component] = json.loads(saved.read_text())[
                    "effective_text"
                ]
                self._proposal_index += 1
                continue
            try:
                result = self.proposer.run_sync(
                    json.dumps(request, ensure_ascii=False),
                    usage_limits=UsageLimits(request_limit=1),
                )
            except ModelAPIError as exc:
                self._service_error = exc
                raise
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
            write_json(
                saved,
                {
                    "component": component,
                    "proposed_text": result.output,
                    "effective_text": text,
                    "validation_error": error,
                    "usage": asdict(result.usage),
                },
            )
            self._proposal_index += 1
        self._selection = None
        return replacements
