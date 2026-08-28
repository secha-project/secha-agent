from collections.abc import Sequence
from typing import Any, cast

import pytest
from pydantic_ai import capture_run_messages
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel

from src.agent.agent import create_agent
from src.agent.capabilities import (
    catalog,
    query,
    semantic_layer,
    table_outputs,
    visualization,
)
from src.agent.deps import Deps
from src.agent.prompt import render_prompt

CATALOG_INSTRUCTIONS = render_prompt(
    "capabilities/catalog/instructions.md",
    frozenset({"catalog"}),
)
QUERY_TEMPLATE = "capabilities/query/instructions.md"
QUERY_STANDALONE = render_prompt(
    QUERY_TEMPLATE,
    frozenset({"query", "table_outputs"}),
)
QUERY_WITH_CATALOG = render_prompt(
    QUERY_TEMPLATE,
    frozenset({"catalog", "query", "table_outputs"}),
)
VISUALIZATION_INSTRUCTIONS = render_prompt(
    "capabilities/visualization/instructions.md",
    frozenset({"query", "table_outputs", "visualization"}),
)


@pytest.mark.parametrize(
    ("selected", "expected_routes"),
    [
        (frozenset({"catalog"}), ["Catalog"]),
        (frozenset({"query", "table_outputs"}), ["Query"]),
        (
            frozenset({"catalog", "query", "table_outputs"}),
            ["Catalog", "Query"],
        ),
        (
            frozenset({"semantic_layer", "table_outputs"}),
            ["Semantic layer"],
        ),
        (
            frozenset({"query", "table_outputs", "visualization"}),
            ["Query", "Visualization"],
        ),
        (
            frozenset({"catalog", "query", "table_outputs", "visualization"}),
            ["Catalog", "Query", "Visualization"],
        ),
        (
            frozenset(
                {
                    "catalog",
                    "query",
                    "semantic_layer",
                    "table_outputs",
                    "visualization",
                }
            ),
            ["Semantic layer", "Catalog", "Query", "Visualization"],
        ),
    ],
)
def test_base_prompt_routes_only_selected_capabilities_in_priority_order(
    selected: frozenset[str],
    expected_routes: list[str],
) -> None:
    prompt = render_prompt("base.md", selected)
    route_lines = [
        line
        for line in prompt.splitlines()
        if line.lstrip("0123456789. ").startswith("**")
    ]

    assert [line.split("**")[1].removesuffix(":") for line in route_lines] == (
        expected_routes
    )
    assert [int(line.split(".", 1)[0]) for line in route_lines] == list(
        range(1, len(expected_routes) + 1)
    )
    for capability_name in ("Semantic layer", "Catalog", "Query", "Visualization"):
        assert (f"**{capability_name}:**" in prompt) == (
            capability_name in expected_routes
        )


@pytest.mark.parametrize(
    "capability_module",
    [semantic_layer, catalog, query, table_outputs, visualization],
)
def test_capabilities_have_descriptions(
    capability_module: Any,
) -> None:
    selected = frozenset(
        {"catalog", "query", "semantic_layer", "table_outputs", "visualization"}
    )
    capability = capability_module.build(selected)

    assert capability.get_description() == capability_module.DESCRIPTION
    assert capability_module.DESCRIPTION


def _static_instructions(capability: Capability[Deps]) -> list[str]:
    instructions = capability.get_instructions()
    assert instructions is not None
    if isinstance(instructions, str):
        return [instructions]
    if not isinstance(instructions, Sequence):
        return []
    return [instruction for instruction in instructions if isinstance(instruction, str)]


def test_catalog_instructions_do_not_depend_on_query() -> None:
    without_query = catalog.build(frozenset({"catalog"}))
    with_query = catalog.build(frozenset({"catalog", "query"}))

    assert _static_instructions(without_query) == [CATALOG_INSTRUCTIONS]
    assert _static_instructions(with_query) == [CATALOG_INSTRUCTIONS]


def test_query_uses_standalone_instructions_without_catalog() -> None:
    capability = query.build(frozenset({"query", "table_outputs"}))

    assert _static_instructions(capability) == [QUERY_STANDALONE]


def test_query_replaces_standalone_instructions_when_catalog_is_selected() -> None:
    capability = query.build(frozenset({"catalog", "query", "table_outputs"}))
    instructions = _static_instructions(capability)

    assert instructions == [QUERY_WITH_CATALOG]
    assert "Resolve uncertain" in instructions[0]
    assert "catalog metadata is unavailable" not in instructions[0]


def test_create_agent_deduplicates_capabilities() -> None:
    model = TestModel(call_tools=[], custom_output_text="done")
    configured_agent = create_agent(
        ("catalog", "catalog", "query", "query", "table_outputs", "table_outputs")
    )

    with capture_run_messages() as messages:
        configured_agent.run_sync(
            "question",
            deps=cast(Deps, cast(Any, object())),
            model=model,
        )

    request = next(message for message in messages if isinstance(message, ModelRequest))
    assert request.instructions is not None
    assert request.instructions.count(CATALOG_INSTRUCTIONS) == 1
    assert request.instructions.count(QUERY_WITH_CATALOG) == 1
    request_parameters = model.last_model_request_parameters
    assert request_parameters is not None
    assert [tool.name for tool in request_parameters.function_tools] == [
        "browse_tables",
        "find_columns",
        "describe_table",
        "query_data",
        "inspect_table_output",
        "display_table_output",
    ]


def test_create_agent_uses_canonical_capability_order() -> None:
    model = TestModel(call_tools=[], custom_output_text="done")
    configured_agent = create_agent(
        (
            "visualization",
            "table_outputs",
            "query",
            "catalog",
            "semantic_layer",
        ),
    )

    with capture_run_messages() as messages:
        configured_agent.run_sync(
            "question",
            deps=cast(Deps, cast(Any, object())),
            model=model,
        )

    request = next(message for message in messages if isinstance(message, ModelRequest))
    assert request.instructions is not None
    headings = [
        "## Semantic layer capability",
        "## Catalog capability",
        "## Query capability",
        "## Table outputs capability",
        "## Visualization capability",
    ]
    assert [request.instructions.index(heading) for heading in headings] == sorted(
        request.instructions.index(heading) for heading in headings
    )

    request_parameters = model.last_model_request_parameters
    assert request_parameters is not None
    assert [tool.name for tool in request_parameters.function_tools] == [
        "list_semantic_models",
        "describe_semantic_model",
        "query_semantic_model",
        "compare_semantic_periods",
        "browse_tables",
        "find_columns",
        "describe_table",
        "query_data",
        "inspect_table_output",
        "display_table_output",
        "visualize_table_output",
    ]


def test_create_agent_rejects_unknown_capabilities() -> None:
    with pytest.raises(ValueError, match=r"Unknown capabilities: \['unknown'\]"):
        create_agent(cast(Any, ("catalog", "unknown")))


def test_visualization_uses_one_instruction_prompt_and_exposes_one_tool() -> None:
    capability = visualization.build(
        frozenset({"query", "table_outputs", "visualization"}),
    )
    assert _static_instructions(capability) == [VISUALIZATION_INSTRUCTIONS]

    model = TestModel(call_tools=[], custom_output_text="done")
    create_agent(("query", "table_outputs", "visualization")).run_sync(
        "question",
        deps=cast(Deps, cast(Any, object())),
        model=model,
    )
    request_parameters = model.last_model_request_parameters
    assert request_parameters is not None
    assert request_parameters.function_tools[-1].name == "visualize_table_output"


def test_semantic_results_can_be_visualized_without_query_or_catalog() -> None:
    model = TestModel(call_tools=[], custom_output_text="done")
    create_agent(("semantic_layer", "table_outputs", "visualization")).run_sync(
        "question",
        deps=cast(Deps, cast(Any, object())),
        model=model,
    )

    request_parameters = model.last_model_request_parameters
    assert request_parameters is not None
    assert [tool.name for tool in request_parameters.function_tools] == [
        "list_semantic_models",
        "describe_semantic_model",
        "query_semantic_model",
        "compare_semantic_periods",
        "inspect_table_output",
        "display_table_output",
        "visualize_table_output",
    ]


@pytest.mark.parametrize(
    ("capabilities", "message"),
    [
        (("visualization",), "requires table_outputs"),
        (("table_outputs", "visualization"), "requires semantic_layer or query"),
        (("query", "visualization"), "requires table_outputs"),
    ],
)
def test_visualization_requires_table_outputs_and_a_producer(
    capabilities: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        create_agent(cast(Any, capabilities))


def test_differently_configured_agents_do_not_contaminate_each_other() -> None:
    standalone_model = TestModel(call_tools=[], custom_output_text="standalone")
    combined_model = TestModel(call_tools=[], custom_output_text="combined")

    with capture_run_messages() as standalone_messages:
        create_agent(("query", "table_outputs")).run_sync(
            "question",
            deps=cast(Deps, cast(Any, object())),
            model=standalone_model,
        )
    with capture_run_messages() as combined_messages:
        create_agent(("catalog", "query", "table_outputs")).run_sync(
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
    assert "catalog metadata is unavailable" in standalone_request.instructions
    assert "Resolve uncertain" not in standalone_request.instructions
    assert "Resolve uncertain" in combined_request.instructions
    assert "catalog metadata is unavailable" not in combined_request.instructions


def test_full_capability_prompt_keeps_tool_mechanics_out_of_instructions() -> None:
    selected = frozenset(
        {"catalog", "query", "semantic_layer", "table_outputs", "visualization"}
    )
    prompt = "\n".join(
        [
            render_prompt("base.md", selected),
            *[
                render_prompt(f"capabilities/{name}/instructions.md", selected)
                for name in (
                    "semantic_layer",
                    "catalog",
                    "query",
                    "table_outputs",
                    "visualization",
                )
            ],
        ]
    )

    assert len(prompt) < 3_500
    assert "VISUALISE" not in prompt
    assert "Call `browse_tables`" not in prompt
    assert "Call `list_semantic_models`" not in prompt
