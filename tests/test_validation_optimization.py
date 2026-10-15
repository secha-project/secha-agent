import json
from dataclasses import replace
from typing import Any, cast
from uuid import uuid4

import pytest

pytest.importorskip("gepa")

import logfire
from gepa import optimize
from pydantic_ai import Agent, ModelSettings, UsageLimits
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from src.agent.config import AgentConfig
from src.agent.deps import Deps
from src.agent.prompt import render_prompt, snapshot_prompts
from src.validation.config import BenchmarkRunConfig
from src.validation.eval_models import (
    BenchmarkCase,
    BenchmarkDataset,
    CaseMetadata,
    TaskCategory,
    TextOutput,
)
from src.validation.evaluators import ValidateTextOutput
from src.validation.optimization import (
    BenchmarkAdapter,
    create_component_selector,
    split_cases,
)
from src.validation.run_optimization import parse_args

SELECTED_CAPABILITIES = frozenset({"query", "table_outputs"})


def make_case(name, query_path="query.sql"):
    return BenchmarkCase(
        name=name,
        inputs=name,
        metadata=CaseMetadata(
            case_id=uuid4(),
            query_path=query_path,
            category=TaskCategory.DATA_DISCOVERY,
        ),
        expected_output=[TextOutput(text=["correct"])],
        evaluators=[ValidateTextOutput()],
    )


def without_trace_metadata(value):
    """Independent runs capture different timestamps and tracing IDs."""
    if isinstance(value, dict):
        return {
            key: without_trace_metadata(
                json.loads(item) if key == "execution" else item
            )
            for key, item in value.items()
            if key not in {"timestamp", "run_id", "conversation_id"}
        }
    if isinstance(value, list):
        return [without_trace_metadata(item) for item in value]
    return value


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "query.sql").write_text(
        "SELECT 'correct'", encoding="utf-8"
    )
    logfire.configure(send_to_logfire=False, console=False)
    monkeypatch.setattr(
        AgentConfig, "experiment_metadata", lambda self: {"model": "offline-test"}
    )
    config = BenchmarkRunConfig(
        agent=AgentConfig(
            _env_file=None,
            model_name="target",
            catalog_url="http://unused",
            catalog_token="unused",
            spark_url="sc://unused",
            otel_exporter_otlp_endpoint="http://unused",
            logfire_service_name="test",
            enabled_capabilities=SELECTED_CAPABILITIES,
        ),
        dataset="test",
        max_concurrency=2,
        task_max_attempts=1,
    )

    def respond(messages, _info):
        request = messages[-1]
        question = request.parts[0].content
        if question == "error":
            raise RuntimeError("Deliberate test failure")
        answer = (
            "correct"
            if "Answer correctly." in request.instructions or question == "pass"
            else "wrong"
        )
        return ModelResponse(parts=[TextPart(answer)])

    return BenchmarkAdapter(
        config=config,
        dataset=BenchmarkDataset(
            name="test", cases=[make_case("train"), make_case("val")]
        ),
        baseline=snapshot_prompts(SELECTED_CAPABILITIES),
        deps=Deps(
            spark=cast(Any, object()),
            catalog=cast(Any, object()),
            schema_name="bronze",
            table_column_page_size=20,
            semantic_layer_enabled=True,
        ),
        model=FunctionModel(respond),
        proposer=Agent(
            TestModel(call_tools=[], custom_output_text="Answer correctly.")
        ),
        run_directory=tmp_path,
    )


def test_adapter_preserves_order_failures_and_candidate_isolation(adapter):
    batch = [
        make_case("error"),
        make_case("pass"),
        make_case("wrong", query_path=None),
    ]
    first = adapter.evaluate(batch, {}, capture_traces=True)
    assert first.scores == [0.0, 1.0, 0.0]
    assert first.outputs == ["", "correct", "wrong"]
    assert first.trajectories[0]["error"] == "RuntimeError: Deliberate test failure"
    assert first.trajectories[1]["attributes"]["agent_messages"]
    assert first.trajectories[1]["reference_sql"] == "SELECT 'correct'"
    assert first.trajectories[2]["reference_sql"] is None
    second = adapter.evaluate(batch, {"base.instructions": "Answer correctly."})
    assert second.scores == [0.0, 1.0, 1.0]
    directories = list((adapter.run_directory / "evaluations").iterdir())
    assert len(directories) == 2
    assert all((directory / "checkpoint.json").exists() for directory in directories)
    assert all((directory / "report.json").exists() for directory in directories)
    assert "Answer correctly." not in adapter.baseline["base.instructions"]


def test_actual_gepa_loop_uses_adapter_and_saves_a_better_candidate(adapter):
    result = optimize(
        seed_candidate={"base.instructions": adapter.baseline["base.instructions"]},
        trainset=[adapter.dataset.cases[0]],
        valset=[adapter.dataset.cases[1]],
        adapter=adapter,
        max_metric_calls=5,
        reflection_minibatch_size=1,
        run_dir=str(adapter.run_directory / "gepa"),
        seed=0,
    )
    assert result.val_aggregate_scores[0] == 0
    assert result.val_aggregate_scores[result.best_idx] == 1
    assert result.best_candidate == {"base.instructions": "Answer correctly."}
    assert list((adapter.run_directory / "proposals").glob("*/proposal.json"))
    json.dumps(result.to_dict())


def test_bad_proposal_retains_existing_component(adapter):
    adapter.proposer = Agent(
        TestModel(call_tools=[], custom_output_text="```text\nWrong format\n```")
    )
    candidate = {"base.instructions": adapter.baseline["base.instructions"]}
    result = adapter.propose_new_texts(
        candidate, {"base.instructions": []}, ["base.instructions"]
    )
    assert result == candidate


def test_feedback_selects_relevant_component_and_passes_diagnosis_to_rewrite(adapter):
    from src.validation.optimization_state import OptimizationProgressLog

    requests = []

    def select(messages, info):
        request = json.loads(messages[-1].parts[0].content)
        requests.append(request)
        assert request["editable_components"] == [
            "base.instructions",
            "query.instructions",
        ]
        assert request["current_prompts"] == adapter.baseline
        assert request["feedback"][0]["feedback"]["case_name"] == "train"
        assert request["feedback"][0]["execution"]
        return ModelResponse(
            parts=[
                TextPart(
                    json.dumps(
                        {
                            "component": "query.instructions",
                            "rationale": "The train trace shows incorrect query behavior; query.instructions governs it.",
                        }
                    )
                )
            ]
        )

    adapter.selector = create_component_selector(
        FunctionModel(select),
        render_prompt("optimization/selector.md", SELECTED_CAPABILITIES),
        ModelSettings(),
    )
    candidate = {
        key: adapter.baseline[key]
        for key in ("base.instructions", "query.instructions")
    }
    log = adapter.run_directory / "progress.log"
    with OptimizationProgressLog(log, "test", adapter.run_directory, 5) as progress:
        adapter.progress = progress
        result = optimize(
            seed_candidate=candidate,
            trainset=[adapter.dataset.cases[0]],
            valset=[adapter.dataset.cases[1]],
            adapter=adapter,
            module_selector=adapter.select_components,
            max_metric_calls=4,
            reflection_minibatch_size=1,
            run_dir=str(adapter.run_directory / "gepa"),
            raise_on_exception=True,
        )
    assert result.best_candidate == candidate | {
        "query.instructions": "Answer correctly."
    }
    assert result.val_aggregate_scores[result.best_idx] == 1
    assert len(requests) == 1
    selection = json.loads(
        (
            adapter.run_directory / "selections/selection-000000/selection.json"
        ).read_text()
    )
    proposal = json.loads(
        (adapter.run_directory / "proposals/proposal-000000/request.json").read_text()
    )
    assert proposal["selection"] == {
        "operation": "selection-000000",
        **selection["selection"],
    }
    assert proposal["component"] == "query.instructions"
    assert (
        "SELECTION" in log.read_text()
        and selection["selection"]["rationale"] in log.read_text()
    )


@pytest.mark.parametrize(
    "invalid",
    [
        {"component": "not-a-component", "rationale": "Wrong choice"},
        {"component": "query_data.description", "rationale": "Active but not editable"},
        {"component": "query.instructions", "rationale": "   "},
        {
            "component": ["base.instructions", "query.instructions"],
            "rationale": "Two edits",
        },
        "not JSON",
    ],
)
def test_selector_retries_invalid_choices_without_silent_fallback(invalid):
    outputs = iter(
        [invalid, {"component": "query.instructions", "rationale": "Fix SQL guidance"}]
    )
    calls = []

    def respond(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart(json.dumps(next(outputs)))])

    selector = create_component_selector(
        FunctionModel(respond), "Select a component.", ModelSettings()
    )
    result = selector.run_sync(
        "Choose using this feedback.",
        deps=("base.instructions", "query.instructions"),
        usage_limits=UsageLimits(request_limit=3),
    )
    assert result.output.component == "query.instructions"
    assert len(calls) == 2


def test_selector_validation_failure_stops_before_rewrite_or_child_evaluation(adapter):
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    calls = []

    def invalid(messages, info):
        calls.append(messages)
        return ModelResponse(
            parts=[TextPart('{"component":"unknown","rationale":"Invalid"}')]
        )

    adapter.selector = create_component_selector(
        FunctionModel(invalid), "Select a component.", ModelSettings()
    )
    with pytest.raises(UnexpectedModelBehavior):
        optimize(
            seed_candidate=dict(adapter.baseline),
            trainset=[adapter.dataset.cases[0]],
            valset=[adapter.dataset.cases[1]],
            adapter=adapter,
            module_selector=adapter.select_components,
            max_metric_calls=5,
            reflection_minibatch_size=1,
            run_dir=str(adapter.run_directory / "gepa"),
            raise_on_exception=True,
        )
    assert len(calls) == 3
    assert not list((adapter.run_directory / "proposals").iterdir())
    assert len(list((adapter.run_directory / "evaluations").iterdir())) == 2
    assert not list((adapter.run_directory / "selections").glob("*/selection.json"))


def test_splits_reject_overlap_and_aliases():
    first, second = make_case("first"), make_case("second")
    dataset = BenchmarkDataset(name="test", cases=[first, second])
    with pytest.raises(ValueError, match="disjoint"):
        split_cases(dataset, ["first"], ["first"])
    with pytest.raises(ValueError, match="Unknown"):
        split_cases(dataset, ["first"], ["missing"])
    second.metadata = first.metadata
    with pytest.raises(ValueError, match="case IDs"):
        split_cases(dataset, ["first"], ["second"])


def test_search_uses_configured_repetitions_and_reflects_each_attempt(adapter):
    adapter = replace(
        adapter, config=adapter.config.model_copy(update={"repetitions": 2})
    )
    answers = iter(["correct", "wrong"])
    adapter.model = FunctionModel(
        lambda messages, info: ModelResponse(parts=[TextPart(next(answers))])
    )
    result = adapter.evaluate([make_case("repeat")], {}, capture_traces=True)
    assert result.scores == [0.5]
    assert len(result.trajectories[0]["attempts"]) == 2
    assert all(
        attempt["reference_sql"] == "SELECT 'correct'"
        for attempt in result.trajectories[0]["attempts"]
    )
    feedback = adapter.make_reflective_dataset({}, result, ["base.instructions"])
    assert sorted(
        (record["feedback"]["actual_output"], record["feedback"]["score"])
        for record in feedback["base.instructions"]
    ) == [("correct", 1), ("wrong", 0)]


def test_cli_allows_component_listing_without_credentials():
    assert parse_args(["--list-components"]).list_components
    with pytest.raises(SystemExit):
        parse_args(
            ["--config", "example", "--train", "first", "--validation", "second"]
        )


def test_resume_uses_run_path_instead_of_new_results_root():
    with pytest.raises(SystemExit):
        parse_args(["--resume", "/saved/run", "--results-dir", "/different/drive"])


@pytest.mark.parametrize(
    ("name", "provider"),
    [
        ("gpt-6-sol", "openai"),
        ("gpt-6-luna", "openai"),
        ("o3", "openai"),
        ("glm-5.3", "zai"),
        ("Qwen3.8-27B-AIC", "tuni"),
    ],
)
def test_reflection_model_name_selects_provider_without_benchmark_config(
    name, provider
):
    from src.validation.run_optimization import reflection_provider

    assert reflection_provider(name) == provider


def test_custom_reflection_model_requires_explicit_provider():
    from src.validation.run_optimization import reflection_provider

    with pytest.raises(ValueError, match="--reflection-provider"):
        reflection_provider("custom-deployment")
    assert reflection_provider("custom-deployment", "tuni") == "tuni"
    for name in ("", " ", " gpt-6-sol"):
        with pytest.raises(ValueError, match="model name"):
            reflection_provider(name, "openai")


def test_reflection_settings_do_not_inherit_target_model_options(adapter):
    from src.agent.config import PricingReference
    from src.validation.run_optimization import bounded_agent, reflection_agent

    args = parse_args(
        [
            "--config",
            "test",
            "--reflection-model",
            "gpt-6-sol",
            "--train",
            "a",
            "--validation",
            "b",
        ]
    )
    target = adapter.config.agent.model_copy(
        update={
            "temperature": 0.7,
            "pricing": PricingReference(model_name="target-pricing"),
        }
    )
    reflection = reflection_agent(bounded_agent(target, args), args)
    assert (
        reflection.model_name == "gpt-6-sol" and reflection.model_provider == "openai"
    )
    assert reflection.temperature is None and reflection.pricing is None
    assert not reflection.enabled_capabilities
    assert reflection.model_request_max_attempts == 20
    assert target.model_name == "target" and target.temperature == 0.7


@pytest.mark.parametrize("dry_run", [True, False])
def test_cli_snapshots_inputs_and_exports_winner(adapter, monkeypatch, dry_run):
    from src.validation import run_optimization as cli

    run_directory = adapter.run_directory / "cli-run"
    run_directory.mkdir()

    def load_config(name):
        assert name == "test", "Only the target uses a benchmark config"
        return name, adapter.config

    monkeypatch.setattr(cli, "load_run_config", load_config)
    monkeypatch.setattr(
        cli, "create_run_directory", lambda *args, **kwargs: run_directory
    )
    source = adapter.run_directory / "source"
    (source / "test").mkdir(parents=True)
    (source / "test" / "dataset.yaml").write_text(
        "name: test\ncases:\n"
        "- name: train\n  inputs: train\n  metadata:\n"
        "    case_id: 11111111-1111-1111-1111-111111111111\n"
        "    category: data discovery\n    query_path: query.sql\n"
        "  expected_output:\n  - text: [correct]\n"
        "- name: val\n  inputs: val\n  metadata:\n"
        "    case_id: 22222222-2222-2222-2222-222222222222\n"
        "    category: data discovery\n    query_path: query.sql\n"
        "  expected_output:\n  - text: [correct]\n"
    )
    (source / "test" / "query.sql").write_text("SELECT 'correct'", encoding="utf-8")
    monkeypatch.setattr(cli, "BENCHMARKS_DIR", source)
    monkeypatch.setattr(cli, "load_dataset", lambda name: adapter.dataset)
    args = ["--config", "test", "--train", "train", "--validation", "val"]
    if dry_run:
        monkeypatch.setattr(
            cli,
            "create_dependencies",
            lambda *args: pytest.fail("No runtime in a dry run"),
        )
        monkeypatch.setattr(
            cli, "optimize", lambda **kwargs: pytest.fail("No optimizer in a dry run")
        )
        args.append("--dry-run")
    else:
        monkeypatch.setattr(cli, "create_dependencies", lambda *args: adapter.deps)
        monkeypatch.setattr(cli, "configure_observability", lambda *args: None)
        monkeypatch.setattr(
            AgentConfig,
            "create_model",
            lambda self: (
                TestModel(call_tools=[], custom_output_text="Answer correctly.")
                if self.model_name == "reflection"
                else adapter.model
            ),
        )
        args.extend(
            [
                "--reflection-model",
                "reflection",
                "--reflection-provider",
                "openai",
                "--components",
                "base.instructions",
                "--max-metric-calls",
                "5",
            ]
        )
    cli.main(args)
    manifest = json.loads((run_directory / "manifest.json").read_text())
    assert "semantic_layer.instructions" not in manifest["prompt_files"]
    assert manifest["benchmark"]["repetitions"] == adapter.config.repetitions
    if dry_run:
        assert manifest["reflection"] is None
        assert set(manifest["components"]) == set(adapter.baseline)
    else:
        best = json.loads((run_directory / "best-prompts.json").read_text())
        assert best == adapter.baseline | {"base.instructions": "Answer correctly."}
        assert (
            run_directory / "prompts" / "base.md"
        ).read_text() == "Answer correctly.\n"
        assert (
            "+Answer correctly.\n"
            in (run_directory / "prompt-changes.diff").read_text()
        )
    assert json.loads((run_directory / "baseline.json").read_text()) == adapter.baseline


@pytest.fixture
def resumable_cli(adapter, monkeypatch, tmp_path):
    from types import SimpleNamespace

    import yaml
    from pydantic_ai.exceptions import ModelAPIError

    from src.validation import run_optimization as cli

    config = adapter.config.model_copy(update={"repetitions": 2, "max_concurrency": 1})
    source = tmp_path / "resume-source"
    (source / "test").mkdir(parents=True)
    cases = [
        make_case(name, query_path=None)
        for name in ("train_a", "train_b", "val_a", "val_b", "heldout")
    ]
    (source / "test" / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "test",
                "cases": [
                    {
                        "name": case.name,
                        "inputs": case.inputs,
                        "metadata": case.metadata.model_dump(mode="json"),
                        "expected_output": [{"text": ["correct"]}],
                    }
                    for case in cases
                ],
            }
        )
    )
    state = SimpleNamespace(
        directory=tmp_path / "run",
        target_calls=[],
        proposer_calls=0,
        selector_calls=0,
        selection_requests=[],
        proposal_requests=[],
        proposal_texts=None,
        selection_components=None,
        target_failures=0,
        proposer_failures=0,
        selector_failures=0,
        config=config,
        model_settings=[],
    )
    state.directory.mkdir()

    def respond(messages, info):
        question = messages[-1].parts[0].content
        assert question in {case.inputs for case in cases}
        state.target_calls.append(question)
        if state.target_failures:
            state.target_failures -= 1
            raise ModelAPIError("offline", "temporary outage")
        answer = (
            "correct" if "Answer correctly." in messages[-1].instructions else "wrong"
        )
        return ModelResponse(parts=[TextPart(answer)])

    def propose(messages, info):
        request = json.loads(messages[-1].parts[0].content)
        if "editable_components" in request:
            state.selector_calls += 1
            state.selection_requests.append(request)
            if state.selector_failures:
                state.selector_failures -= 1
                raise ModelAPIError("reflection", "temporary selection outage")
            return ModelResponse(
                parts=[
                    TextPart(
                        json.dumps(
                            {
                                "component": (
                                    state.selection_components[state.selector_calls - 1]
                                    if state.selection_components
                                    else "query.instructions"
                                ),
                                "rationale": "Training feedback points to query instructions.",
                            }
                        )
                    )
                ]
            )
        state.proposer_calls += 1
        state.proposal_requests.append(request)
        if state.proposer_failures:
            state.proposer_failures -= 1
            raise ModelAPIError("offline", "temporary reflection outage")
        text = (
            state.proposal_texts[state.proposer_calls - 1]
            if state.proposal_texts
            else "Answer correctly."
        )
        return ModelResponse(parts=[TextPart(text)])

    def load_config(name):
        assert name == "test", "Only the target uses a benchmark config"
        return name, state.config

    def model(settings):
        state.model_settings.append(settings.model_request_max_attempts)
        return FunctionModel(
            propose if settings.model_name == "reflection" else respond
        )

    monkeypatch.setattr(cli, "load_run_config", load_config)
    monkeypatch.setattr(
        cli, "create_run_directory", lambda *args, **kwargs: state.directory
    )
    monkeypatch.setattr(cli, "BENCHMARKS_DIR", source)
    monkeypatch.setattr(
        cli, "load_dataset", lambda _: BenchmarkDataset(name="test", cases=cases)
    )
    monkeypatch.setattr(cli, "create_dependencies", lambda _: adapter.deps)
    monkeypatch.setattr(cli, "configure_observability", lambda _: None)
    monkeypatch.setattr(AgentConfig, "create_model", model)
    state.cli = cli
    state.args = [
        "--config",
        "test",
        "--reflection-model",
        "reflection",
        "--reflection-provider",
        "openai",
        "--train",
        "train_a",
        "train_b",
        "--validation",
        "val_a",
        "val_b",
        "--test",
        "heldout",
        "--components",
        "base.instructions",
        "--max-metric-calls",
        "12",
        "--reflection-minibatch-size",
        "1",
        "--max-resumes",
        "0",
    ]
    return state


@pytest.mark.parametrize("separate_progress_log", [False, True])
def test_custom_results_root_contains_logs_artifacts_and_resumes(
    resumable_cli, monkeypatch, tmp_path, separate_progress_log
):
    from pathlib import Path

    import pyarrow as pa

    from src.agent.outputs.artifacts import TableArtifact
    from src.validation import runner

    state = resumable_cli
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "large drive" / "optimizer results"
    default_root = tmp_path / "unused-default-results"
    monkeypatch.setattr(runner, "RESULTS_DIR", default_root)
    monkeypatch.setattr(state.cli, "create_run_directory", runner.create_run_directory)
    table = pa.table({"answer": ["correct"]})
    buffer = pa.BufferOutputStream()
    with pa.ipc.new_stream(buffer, table.schema) as stream:
        stream.write_table(table)
    artifact = TableArtifact(
        arrow_bytes=buffer.getvalue().to_pybytes(), truncated=False
    )
    write_artifacts = runner.write_evaluation_artifacts
    artifact_paths = []

    def publish_artifact(_artifacts):
        paths = write_artifacts((artifact,))
        artifact_paths.extend(paths)
        return paths

    monkeypatch.setattr(runner, "write_evaluation_artifacts", publish_artifact)
    original_evaluate = BenchmarkAdapter.evaluate
    interrupted = False

    def evaluate_and_interrupt(self, *args, **kwargs):
        nonlocal interrupted
        result = original_evaluate(self, *args, **kwargs)
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(BenchmarkAdapter, "evaluate", evaluate_and_interrupt)
    args = [*state.args, "--results-dir", str(root.relative_to(tmp_path))]
    separate_log = tmp_path / "summary logs" / "optimizer.log"
    if separate_progress_log:
        args += ["--progress-log", str(separate_log)]
    with pytest.raises(KeyboardInterrupt):
        state.cli.main(args)
    (directory,) = (root / "test").iterdir()
    manifest = json.loads((directory / "manifest.json").read_text())
    log = separate_log if separate_progress_log else directory / "progress.log"
    assert Path(manifest["progress_log"]) == log
    state.cli.main(["--resume", str(directory)])
    assert len(state.target_calls) == 28  # Completed initial attempts were reused.
    for relative in (
        "result.json",
        "held-out-comparison.json",
        "best-prompts.json",
        "gepa/run_log.txt",
        "gepa/run_log_stderr.txt",
        "gepa/gepa_state.bin",
    ):
        assert (directory / relative).is_file()
    assert artifact_paths
    assert all(
        path.is_relative_to(directory) and path.is_file() for path in artifact_paths
    )
    assert not default_root.exists()
    assert "INTERRUPTED" in log.read_text() and "FINISHED" in log.read_text()
    # A subsequent new run on the same drive must not overwrite the existing run.
    state.cli.main([*args, "--dry-run"])
    assert len(list((root / "test").iterdir())) == 2
    assert json.loads((directory / "manifest.json").read_text()) == manifest


def test_invalid_results_root_fails_before_model_calls(
    resumable_cli, monkeypatch, tmp_path
):
    from src.validation import runner

    state = resumable_cli
    root = tmp_path / "not a directory"
    root.write_text("existing file")
    monkeypatch.setattr(state.cli, "create_run_directory", runner.create_run_directory)
    with pytest.raises(NotADirectoryError):
        state.cli.main([*state.args, "--results-dir", str(root)])
    assert not state.target_calls and not state.proposer_calls
    assert root.read_text() == "existing file"


@pytest.mark.parametrize("multiple_components", [False, True])
@pytest.mark.parametrize("interrupt_after", [1, 3, 5, "selection", "proposal"])
def test_cli_resume_replays_saved_operations_and_restores_sampling(
    resumable_cli, monkeypatch, interrupt_after, multiple_components
):
    state = resumable_cli
    if multiple_components:
        state.args.insert(
            state.args.index("base.instructions") + 1, "query.instructions"
        )
    state.cli.main(state.args)
    expected = json.loads((state.directory / "result.json").read_text())
    expected_calls = list(state.target_calls)
    expected_proposals = state.proposer_calls
    expected_selections = state.selector_calls
    expected_selection_requests = list(state.selection_requests)
    expected_proposal_requests = list(state.proposal_requests)
    state.directory = state.directory.with_name("interrupted")
    state.directory.mkdir()
    state.target_calls.clear()
    state.proposer_calls = 0
    state.selector_calls = 0
    state.selection_requests.clear()
    state.proposal_requests.clear()
    method = {"selection": "select_components", "proposal": "_propose_new_texts"}.get(
        interrupt_after, "evaluate"
    )
    original_operation = getattr(BenchmarkAdapter, method)
    interrupted = False

    def operate_and_interrupt(self, *args, **kwargs):
        nonlocal interrupted
        result = original_operation(self, *args, **kwargs)
        if not interrupted and (
            method != "evaluate" or self._evaluation_index == interrupt_after
        ):
            interrupted = True
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(BenchmarkAdapter, method, operate_and_interrupt)
    with pytest.raises(KeyboardInterrupt):
        state.cli.main(state.args)
    state.cli.main(["--resume", str(state.directory), "--max-resumes", "0"])
    resumed = json.loads((state.directory / "result.json").read_text())
    for key in (
        "candidates",
        "parents",
        "val_aggregate_scores",
        "val_subscores",
        "total_metric_calls",
    ):
        assert resumed[key] == expected[key]
    assert state.target_calls == expected_calls
    assert state.proposer_calls == expected_proposals
    assert state.selector_calls == expected_selections
    assert without_trace_metadata(state.selection_requests) == without_trace_metadata(
        expected_selection_requests
    )
    assert without_trace_metadata(state.proposal_requests) == without_trace_metadata(
        expected_proposal_requests
    )
    assert state.selector_calls == (1 if multiple_components else 0)
    for request in state.selection_requests:
        assert {
            record["feedback"]["case_name"].split(" [")[0]
            for record in request["feedback"]
        } <= {"train_a", "train_b"}
    assert set(state.model_settings) == {20}
    assert "INTERRUPTED" in (state.directory / "progress.log").read_text()
    assert "FINISHED" in (state.directory / "progress.log").read_text()


@pytest.mark.parametrize(
    ("failure", "multiple_components"),
    [
        ("target", False),
        ("target", True),
        ("proposer", False),
        ("proposer", True),
        ("selector", True),
    ],
)
def test_cli_automatically_resumes_model_service_failures(
    resumable_cli, failure, multiple_components
):
    state = resumable_cli
    if multiple_components:
        state.args.insert(
            state.args.index("base.instructions") + 1, "query.instructions"
        )
    setattr(state, f"{failure}_failures", 1)
    args = state.args[:-2] + ["--max-resumes", "1", "--resume-delay-seconds", "0"]
    state.cli.main(args)
    result = json.loads((state.directory / "result.json").read_text())
    assert max(result["val_aggregate_scores"]) == 1
    assert result["total_metric_calls"] == 12
    log = (state.directory / "progress.log").read_text()
    assert "PAUSED" in log and "RESUMING" in log and "FINISHED" in log
    assert "reflection_model=reflection" in log
    assert "task_model=target" in log
    assert "component_selection=feedback" in log
    assert state.selector_calls == (
        1 + (failure == "selector") if multiple_components else 0
    )
    assert state.proposer_calls == 1 + (failure == "proposer")
    if failure == "proposer":
        # GEPA retries its custom proposer internally; this must not send another
        # request before the coordinator resumes from the pre-iteration state.
        assert state.proposer_calls == 2
    assert len(state.target_calls) == 28 + (failure == "target")
    checkpoint = json.loads(
        (
            state.directory / "evaluations" / "search-000000" / "checkpoint.json"
        ).read_text()
    )
    assert len(checkpoint["cases"]) == 4
    assert not checkpoint["failures"]


def test_resume_restores_selection_cursor_after_rejected_edit(
    resumable_cli, monkeypatch
):
    state = resumable_cli
    state.args.insert(state.args.index("base.instructions") + 1, "query.instructions")
    state.proposal_texts = ["An unhelpful edit.", "Answer correctly."]
    state.selection_components = ["query.instructions", "base.instructions"]
    state.cli.main(state.args)
    expected = json.loads((state.directory / "result.json").read_text())
    expected_calls = list(state.target_calls)
    state.directory = state.directory.with_name("second-selection-interrupted")
    state.directory.mkdir()
    state.target_calls.clear()
    state.proposer_calls = state.selector_calls = 0
    original = BenchmarkAdapter.select_components
    interrupted = False

    def select_and_interrupt(self, *args, **kwargs):
        nonlocal interrupted
        result = original(self, *args, **kwargs)
        if not interrupted and self._selection_index == 2:
            interrupted = True
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(BenchmarkAdapter, "select_components", select_and_interrupt)
    with pytest.raises(KeyboardInterrupt):
        state.cli.main(state.args)
    state.cli.main(["--resume", str(state.directory)])
    resumed = json.loads((state.directory / "result.json").read_text())
    assert resumed["candidates"] == expected["candidates"]
    assert resumed["total_metric_calls"] == expected["total_metric_calls"]
    assert state.target_calls == expected_calls
    assert state.selector_calls == state.proposer_calls == 2
    assert len(list((state.directory / "selections").glob("*/selection.json"))) == 2
    request = json.loads(
        (state.directory / "proposals/proposal-000001/request.json").read_text()
    )
    assert request["selection"]["operation"] == "selection-000001"
    assert request["component"] == "base.instructions"


def test_resume_finishes_heldout_without_restarting_search(resumable_cli, monkeypatch):
    state = resumable_cli
    original_evaluate = BenchmarkAdapter.evaluate
    interrupted = False

    def evaluate_and_interrupt(self, *args, **kwargs):
        nonlocal interrupted
        result = original_evaluate(self, *args, **kwargs)
        if not interrupted and self.evaluation_phase == "held-out-baseline":
            interrupted = True
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(BenchmarkAdapter, "evaluate", evaluate_and_interrupt)
    with pytest.raises(KeyboardInterrupt):
        state.cli.main(state.args)
    assert (state.directory / "best-prompts.json").exists()
    previous_calls = len(state.target_calls)
    monkeypatch.setattr(
        state.cli, "optimize", lambda **kwargs: pytest.fail("Search already finished")
    )
    state.cli.main(["--resume", str(state.directory)])
    assert len(state.target_calls) == previous_calls + 2
    heldout = json.loads((state.directory / "held-out-comparison.json").read_text())
    assert heldout["baseline_scores"] == [0] and heldout["best_scores"] == [1]
    monkeypatch.setattr(
        state.cli,
        "create_dependencies",
        lambda _: pytest.fail("Completed resume must be offline"),
    )
    state.cli.main(["--resume", str(state.directory)])
    assert len(state.target_calls) == previous_calls + 2


@pytest.mark.parametrize(
    "changed",
    [
        "dataset",
        "baseline",
        "selector",
        "selection_policy",
        "config",
        "source",
        "dependencies",
        "old_format",
    ],
)
def test_resume_rejects_changed_inputs_before_external_calls(
    resumable_cli, monkeypatch, changed
):
    state = resumable_cli
    state.cli.main([*state.args, "--dry-run"])
    if changed == "dataset":
        with (state.directory / "dataset" / "dataset.yaml").open("a") as stream:
            stream.write("\n# changed\n")
    elif changed == "baseline":
        with (state.directory / "baseline.json").open("a") as stream:
            stream.write(" ")
    elif changed == "selector":
        with (state.directory / "selector-instructions.md").open("a") as stream:
            stream.write("Changed selection instructions")
    elif changed in {"selection_policy", "old_format"}:
        path = state.directory / "manifest.json"
        manifest = json.loads(path.read_text())
        if changed == "selection_policy":
            manifest["component_selection"] = "round_robin"
        else:
            manifest["format_version"] = 2
        path.write_text(json.dumps(manifest))
    elif changed == "config":
        state.config = state.config.model_copy(update={"repetitions": 3})
    elif changed == "source":
        monkeypatch.setattr(
            state.cli, "source_hashes", lambda: {"changed.py": "changed"}
        )
    else:
        monkeypatch.setattr(
            state.cli, "dependency_versions", lambda: {"gepa": "changed"}
        )
    monkeypatch.setattr(
        state.cli, "create_dependencies", lambda _: pytest.fail("No external call")
    )
    with pytest.raises(
        ValueError, match="Cannot resume|incompatible optimization format"
    ):
        state.cli.main(["--resume", str(state.directory)])


def test_resume_rejects_operation_mismatch(adapter):
    batch = [make_case("pass")]
    adapter.evaluate(batch, {})
    adapter._evaluation_index = 0
    with pytest.raises(ValueError, match="saved inputs"):
        adapter.evaluate(batch, {"base.instructions": "Different candidate"})


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--reflection-model", "gpt-6-sol"),
        ("--reflection-provider", "zai"),
    ],
)
def test_resume_rejects_changed_reflection_model_or_provider(
    resumable_cli, monkeypatch, flag, value
):
    state = resumable_cli
    state.cli.main([*state.args, "--dry-run"])
    manifest = json.loads((state.directory / "manifest.json").read_text())
    assert manifest["reflection_model"] == "reflection"
    assert manifest["reflection_provider"] == "openai"
    assert "reflection_config" not in manifest
    monkeypatch.setattr(
        state.cli, "create_dependencies", lambda _: pytest.fail("No external calls")
    )
    with pytest.raises(ValueError, match="Resume reflection_"):
        state.cli.main(["--resume", str(state.directory), flag, value])


def test_resume_limits_stop_and_leave_manual_resume_available(resumable_cli):
    from src.validation.optimization_state import OptimizationServiceError

    state = resumable_cli
    state.target_failures = 100
    with pytest.raises(OptimizationServiceError):
        state.cli.main(
            state.args[:-2] + ["--max-resumes", "1", "--resume-delay-seconds", "0"]
        )
    assert (state.directory / "progress.log").read_text().count("RESUMING") == 1
    assert not (state.directory / "result.json").exists()
    state.target_failures = 0
    state.cli.main(["--resume", str(state.directory), "--max-resumes", "0"])
    assert (state.directory / "held-out-comparison.json").exists()


def test_judge_api_failure_is_left_pending_instead_of_scored(adapter):
    from pydantic_ai.exceptions import ModelAPIError
    from pydantic_evals.evaluators import Evaluator

    from src.validation.optimization_state import OptimizationServiceError

    class UnavailableJudge(Evaluator):
        def evaluate(self, ctx):
            raise ModelAPIError("judge", "temporary judge outage")

    case = make_case("pass")
    case.evaluators = [*case.evaluators, UnavailableJudge()]
    with pytest.raises(OptimizationServiceError):
        adapter.evaluate([case], {})
    checkpoint = json.loads(
        (
            adapter.run_directory / "evaluations" / "search-000000" / "checkpoint.json"
        ).read_text()
    )
    assert checkpoint["cases"] == [] and checkpoint["failures"] == []
