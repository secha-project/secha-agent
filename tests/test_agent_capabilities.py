from collections.abc import Sequence
from typing import Any, cast

import pytest
from pydantic_ai import capture_run_messages
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel

from src.agent.agent import (
    build_catalog,
    build_query,
    build_visualization,
    compose_prompts,
    create_agent,
    load_prompt,
)
from src.agent.deps import Deps

CATALOG_INSTRUCTIONS = load_prompt("capabilities/catalog/instructions.md")
QUERY_CORE = load_prompt("capabilities/query/core.md")
QUERY_STANDALONE = load_prompt("capabilities/query/standalone.md")
QUERY_WITH_CATALOG = load_prompt("capabilities/query/with_catalog.md")
VISUALIZATION_INSTRUCTIONS = load_prompt("capabilities/visualization/instructions.md")


def _static_instructions(capability: Capability[Deps]) -> list[str]:
    instructions = capability.get_instructions()
    assert instructions is not None
    if isinstance(instructions, str):
        return [instructions]
    if not isinstance(instructions, Sequence):
        return []
    return [instruction for instruction in instructions if isinstance(instruction, str)]


def test_catalog_instructions_do_not_depend_on_query() -> None:
    without_query = build_catalog(frozenset({"catalog"}))
    with_query = build_catalog(frozenset({"catalog", "query"}))

    assert _static_instructions(without_query) == [CATALOG_INSTRUCTIONS]
    assert _static_instructions(with_query) == [CATALOG_INSTRUCTIONS]


def test_query_uses_standalone_instructions_without_catalog() -> None:
    capability = build_query(frozenset({"query"}))

    assert _static_instructions(capability) == [
        compose_prompts(
            "capabilities/query/core.md",
            "capabilities/query/standalone.md",
        )
    ]


def test_query_replaces_standalone_instructions_when_catalog_is_selected() -> None:
    capability = build_query(frozenset({"catalog", "query"}))
    instructions = _static_instructions(capability)

    assert instructions == [
        compose_prompts(
            "capabilities/query/core.md",
            "capabilities/query/with_catalog.md",
        )
    ]
    assert QUERY_CORE in instructions[0]
    assert QUERY_WITH_CATALOG in instructions[0]
    assert QUERY_STANDALONE not in instructions[0]


def test_create_agent_deduplicates_capabilities() -> None:
    model = TestModel(call_tools=[], custom_output_text="done")
    configured_agent = create_agent(("catalog", "catalog", "query", "query"))

    with capture_run_messages() as messages:
        configured_agent.run_sync(
            "question",
            deps=cast(Deps, cast(Any, object())),
            model=model,
        )

    request = next(message for message in messages if isinstance(message, ModelRequest))
    assert request.instructions is not None
    assert request.instructions.count(CATALOG_INSTRUCTIONS) == 1
    assert request.instructions.count(QUERY_CORE) == 1
    request_parameters = model.last_model_request_parameters
    assert request_parameters is not None
    assert [tool.name for tool in request_parameters.function_tools] == [
        "browse_tables",
        "find_columns",
        "describe_table",
        "query_data",
        "inspect_query_result",
        "display_query_result",
    ]


def test_create_agent_rejects_unknown_capabilities() -> None:
    with pytest.raises(ValueError, match=r"Unknown capabilities: \['unknown'\]"):
        create_agent(cast(Any, ("catalog", "unknown")))


def test_visualization_uses_one_instruction_prompt_and_exposes_one_tool() -> None:
    capability = build_visualization(frozenset({"catalog", "query", "visualization"}))
    assert _static_instructions(capability) == [VISUALIZATION_INSTRUCTIONS]

    model = TestModel(call_tools=[], custom_output_text="done")
    create_agent(("catalog", "query", "visualization")).run_sync(
        "question",
        deps=cast(Deps, cast(Any, object())),
        model=model,
    )
    request_parameters = model.last_model_request_parameters
    assert request_parameters is not None
    assert request_parameters.function_tools[-1].name == "visualize_query_result"


@pytest.mark.parametrize(
    "capabilities",
    [
        ("visualization",),
        ("catalog", "visualization"),
        ("query", "visualization"),
    ],
)
def test_visualization_requires_catalog_and_query(
    capabilities: tuple[str, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match="visualization capability requires catalog and query",
    ):
        create_agent(cast(Any, capabilities))


def test_differently_configured_agents_do_not_contaminate_each_other() -> None:
    standalone_model = TestModel(call_tools=[], custom_output_text="standalone")
    combined_model = TestModel(call_tools=[], custom_output_text="combined")

    with capture_run_messages() as standalone_messages:
        create_agent(("query",)).run_sync(
            "question",
            deps=cast(Deps, cast(Any, object())),
            model=standalone_model,
        )
    with capture_run_messages() as combined_messages:
        create_agent(("catalog", "query")).run_sync(
            "question",
            deps=cast(Deps, cast(Any, object())),
            model=combined_model,
        )

    standalone_request = next(
        message for message in standalone_messages if isinstance(message, ModelRequest)
    )
    combined_request = next(
        message for message in combined_messages if isinstance(message, ModelRequest)
    )
    assert standalone_request.instructions is not None
    assert combined_request.instructions is not None
    assert QUERY_STANDALONE in standalone_request.instructions
    assert QUERY_WITH_CATALOG not in standalone_request.instructions
    assert QUERY_WITH_CATALOG in combined_request.instructions
    assert QUERY_STANDALONE not in combined_request.instructions
