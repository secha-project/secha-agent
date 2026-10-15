from typing import Any, cast

import pytest
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel

from src.agent.agent import create_agent
from src.agent.capabilities import CAPABILITIES
from src.agent.deps import Deps
from src.agent.prompt import PROMPT_ROOT
from src.optimization import prompt

ALL = frozenset(capability.NAME for capability in CAPABILITIES)


def capture(prompts=None, capabilities=ALL):
    model = TestModel(call_tools=[], custom_output_text="done")
    overrides = (
        prompt.to_prompt_overrides(prompts, capabilities)
        if prompts is not None
        else None
    )
    with capture_run_messages() as messages:
        create_agent(capabilities, prompt_overrides=overrides).run_sync(
            "question", model=model, deps=cast(Deps, cast(Any, object()))
        )
    request = next(message for message in messages if isinstance(message, ModelRequest))
    return request.instructions, model.last_model_request_parameters


@pytest.mark.parametrize("capabilities", [ALL, frozenset({"query", "table_outputs"})])
def test_snapshot_preserves_entire_model_context(capabilities):
    original = capture(capabilities=capabilities)
    baseline = prompt.snapshot_prompts(capabilities)
    assert all("{%" not in text for text in baseline.values())
    assert capture(baseline, capabilities) == original


def test_every_discovered_component_reaches_the_model():
    files = prompt.prompt_files(ALL)
    assert set(files.values()) == {
        path.relative_to(PROMPT_ROOT).as_posix() for path in PROMPT_ROOT.rglob("*.md")
    }
    prompts = {key: f"Marker for {key}." for key in files}
    instructions, parameters = capture(prompts)
    for key, text in prompts.items():
        if key.endswith(".instructions"):
            assert text in instructions
        else:
            name = key.removesuffix(".description")
            assert (
                next(
                    tool for tool in parameters.function_tools if tool.name == name
                ).description
                == text
            )


@pytest.mark.parametrize("key", ["typo.instructions", "query.instructions"])
def test_unknown_or_inactive_component_is_rejected(key):
    with pytest.raises(ValueError, match="Unknown or inactive"):
        prompt.to_prompt_overrides({key: "text"}, frozenset({"catalog"}))


def test_discovers_new_tool_prompts_only_for_selected_capabilities(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(prompt, "PROMPT_ROOT", tmp_path)
    directory = tmp_path / "capabilities" / "catalog" / "tools"
    directory.mkdir(parents=True)
    (directory / "new_tool.md").write_text("New tool description.")
    assert prompt.to_prompt_overrides(
        {"new_tool.description": "Replacement."}, frozenset({"catalog"})
    ) == {"capabilities/catalog/tools/new_tool.md": "Replacement."}
    assert "new_tool.description" not in prompt.prompt_files(frozenset())


def test_duplicate_tool_names_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt, "PROMPT_ROOT", tmp_path)
    for capability in ("catalog", "query"):
        directory = tmp_path / "capabilities" / capability / "tools"
        directory.mkdir(parents=True)
        (directory / "same_tool.md").write_text("Tool description.")
    with pytest.raises(
        ValueError, match="Duplicate prompt component: same_tool.description"
    ):
        prompt.prompt_files(frozenset({"catalog", "query"}))
