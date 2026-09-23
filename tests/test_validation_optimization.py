import json
from dataclasses import replace
from typing import Any, cast
from uuid import uuid4

import pytest

pytest.importorskip("gepa")

import logfire
from gepa import optimize
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from src.agent.config import AgentConfig
from src.agent.deps import Deps
from src.agent.prompt import snapshot_prompts
from src.validation.config import BenchmarkRunConfig
from src.validation.eval_models import (
    BenchmarkCase,
    BenchmarkDataset,
    CaseMetadata,
    TaskCategory,
    TextOutput,
)
from src.validation.evaluators import ValidateTextOutput
from src.validation.optimization import BenchmarkAdapter, split_cases
from src.validation.run_optimization import parse_args

SELECTED_CAPABILITIES = frozenset({"query", "table_outputs"})


def make_case(name):
    return BenchmarkCase(
        name=name,
        inputs=name,
        metadata=CaseMetadata(
            case_id=uuid4(),
            query_code="SELECT 'correct'",
            category=TaskCategory.DATA_DISCOVERY,
        ),
        expected_output=[TextOutput(text=["correct"])],
        evaluators=[ValidateTextOutput()],
    )


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    logfire.configure(send_to_logfire=False, console=False)
    monkeypatch.setattr(
        AgentConfig, "experiment_metadata", lambda self: {"model": "offline-test"}
    )
    config = BenchmarkRunConfig(
        agent=AgentConfig(
            _env_file=None,
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
    batch = [make_case("error"), make_case("pass"), make_case("wrong")]
    first = adapter.evaluate(batch, {}, capture_traces=True)
    assert first.scores == [0.0, 1.0, 0.0]
    assert first.outputs == ["", "correct", "wrong"]
    assert first.trajectories[0]["error"] == "RuntimeError: Deliberate test failure"
    assert first.trajectories[1]["attributes"]["agent_messages"]
    assert first.trajectories[1]["reference_sql"] == "SELECT 'correct'"
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


@pytest.mark.parametrize("dry_run", [True, False])
def test_cli_snapshots_inputs_and_exports_winner(adapter, monkeypatch, dry_run):
    from src.validation import run_optimization as cli

    reflection_config = adapter.config.model_copy(
        update={
            "agent": adapter.config.agent.model_copy(
                update={"model_name": "reflection"}
            ),
        }
    )
    monkeypatch.setattr(
        cli,
        "load_run_config",
        lambda name: (
            name,
            reflection_config if name == "reflection" else adapter.config,
        ),
    )
    monkeypatch.setattr(
        cli, "create_run_directory", lambda *args: adapter.run_directory
    )
    source = adapter.run_directory / "source"
    (source / "test").mkdir(parents=True)
    (source / "test" / "dataset.yaml").write_text(
        "name: test\ncases:\n"
        "- name: train\n  inputs: train\n  metadata:\n"
        "    case_id: 11111111-1111-1111-1111-111111111111\n"
        "    category: data discovery\n    query_code: ''\n"
        "  expected_output:\n  - text: [correct]\n"
        "- name: val\n  inputs: val\n  metadata:\n"
        "    case_id: 22222222-2222-2222-2222-222222222222\n"
        "    category: data discovery\n    query_code: ''\n"
        "  expected_output:\n  - text: [correct]\n"
    )
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
                if self is reflection_config.agent
                else adapter.model
            ),
        )
        args.extend(
            [
                "--reflection-config",
                "reflection",
                "--components",
                "base.instructions",
                "--max-metric-calls",
                "5",
            ]
        )
    cli.main(args)
    manifest = json.loads((adapter.run_directory / "manifest.json").read_text())
    assert "semantic_layer.instructions" not in manifest["prompt_files"]
    assert manifest["benchmark"]["repetitions"] == adapter.config.repetitions
    if dry_run:
        assert manifest["reflection"] is None
        assert set(manifest["components"]) == set(adapter.baseline)
    else:
        best = json.loads((adapter.run_directory / "best-prompts.json").read_text())
        assert best == adapter.baseline | {"base.instructions": "Answer correctly."}
        assert (
            adapter.run_directory / "prompts" / "base.md"
        ).read_text() == "Answer correctly.\n"
        assert (
            "+Answer correctly.\n"
            in (adapter.run_directory / "prompt-changes.diff").read_text()
        )
    assert (
        json.loads((adapter.run_directory / "baseline.json").read_text())
        == adapter.baseline
    )
