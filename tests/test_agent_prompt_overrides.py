from typing import Any, cast

import pytest
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel

from src.agent.agent import create_agent
from src.agent.capabilities import CAPABILITIES
from src.agent.deps import Deps
from src.agent.prompt import PROMPT_FILES, render_prompt, snapshot_prompts

ALL = frozenset(capability.NAME for capability in CAPABILITIES)


def capture(overrides=None, capabilities=ALL):
    model = TestModel(call_tools=[], custom_output_text="done")
    with capture_run_messages() as messages:
        create_agent(capabilities, prompt_overrides=overrides).run_sync(
            "question",
            model=model,
            deps=cast(Deps, cast(Any, object())),
        )
    request = next(message for message in messages if isinstance(message, ModelRequest))
    return request.instructions, model.last_model_request_parameters


@pytest.mark.parametrize("capabilities", [ALL, frozenset({"query", "table_outputs"})])
def test_snapshot_preserves_entire_model_context(capabilities):
    original_instructions, original_parameters = capture(capabilities=capabilities)
    baseline = snapshot_prompts(capabilities)
    assert all("{%" not in text for text in baseline.values())
    snapshot_instructions, snapshot_parameters = capture(baseline, capabilities)
    assert snapshot_instructions == original_instructions
    assert snapshot_parameters == original_parameters


def test_overrides_are_isolated_and_preserve_argument_schemas():
    original_instructions, original_parameters = capture()
    changed_instructions, changed_parameters = capture(
        {
            "query.instructions": "Query policy experiment.",
            "query_data.description": "SQL tool description experiment.",
        }
    )
    assert "Query policy experiment." in changed_instructions
    original = {tool.name: tool for tool in original_parameters.function_tools}
    changed = {tool.name: tool for tool in changed_parameters.function_tools}
    assert changed["query_data"].description == "SQL tool description experiment."
    for name, tool in original.items():
        assert changed[name].parameters_json_schema == tool.parameters_json_schema
        if name != "query_data":
            assert changed[name] == tool
    assert capture()[0] == original_instructions


def test_every_registered_component_reaches_the_model():
    overrides = {key: f"Marker for {key}." for key in PROMPT_FILES}
    instructions, parameters = capture(overrides)
    for key, text in overrides.items():
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


def test_overrides_are_final_text_and_are_not_rendered_again():
    path = PROMPT_FILES["query.instructions"]
    text = '{% if "catalog" in capabilities %}Discover first.{% else %}Use known IDs.{% endif %}'
    overrides = {"query.instructions": text}
    assert render_prompt(path, ALL, overrides=overrides) == text
    assert (
        render_prompt(path, frozenset({"query", "table_outputs"}), overrides=overrides)
        == text
    )
    assert (
        render_prompt(path, ALL, overrides={"query.instructions": "{{ literal }}"})
        == "{{ literal }}"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"typo.instructions": "text"},
        {"query.instructions": ""},
        {"query.instructions": "   "},
    ],
)
def test_invalid_override_fails_before_execution(overrides):
    with pytest.raises(ValueError):
        create_agent(ALL, prompt_overrides=overrides)


def test_inactive_component_is_rejected():
    with pytest.raises(ValueError, match="inactive"):
        create_agent({"catalog"}, prompt_overrides={"query.instructions": "text"})


@pytest.mark.parametrize("semantic", [False, True])
def test_few_shot_changes_examples_without_changing_tools_or_domain_access(semantic):
    capabilities = frozenset({"catalog", "query", "table_outputs", "visualization"})
    if semantic:
        capabilities |= {"semantic_layer"}
    original_instructions, original_parameters = capture(capabilities=capabilities)
    example_instructions, example_parameters = capture(
        capabilities=capabilities | {"few_shot"}
    )

    assert "## Tool-use examples" not in original_instructions
    assert "## Tool-use examples" in example_instructions
    assert ("## Semantic layer capability" in example_instructions) == semantic
    original_tools = {tool.name: tool for tool in original_parameters.function_tools}
    example_tools = {tool.name: tool for tool in example_parameters.function_tools}
    assert original_tools.keys() == example_tools.keys()
    for name, original in original_tools.items():
        example = example_tools[name]
        assert example.parameters_json_schema == original.parameters_json_schema
        assert "Examples:" not in original.description
        assert example.description.startswith(original.description)
        assert "Examples:" in example.description

    # Rendering the example profile must not contaminate cached baseline prompts.
    assert capture(capabilities=capabilities) == (
        original_instructions,
        original_parameters,
    )


def test_few_shot_tool_examples_are_shared_across_semantic_profiles():
    capabilities = frozenset(
        {"catalog", "query", "table_outputs", "visualization", "few_shot"}
    )
    _, raw = capture(capabilities=capabilities)
    _, semantic = capture(capabilities=capabilities | {"semantic_layer"})
    assert raw.function_tools == semantic.function_tools
