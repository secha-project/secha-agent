from typing import Any, cast

import pytest
from pydantic_ai.models.test import TestModel

from src.agent.agent import create_agent
from src.agent.deps import Deps


def test_semantic_capability_requires_native_catalog_query_stack() -> None:
    with pytest.raises(
        ValueError,
        match="semantic_layer capability requires catalog, query, and table_outputs",
    ):
        create_agent(("semantic_layer",))


def test_semantic_capability_adds_policy_without_semantic_tools() -> None:
    model = TestModel(call_tools=[], custom_output_text="done")
    create_agent(
        ("semantic_layer", "catalog", "query", "table_outputs")
    ).run_sync(
        "question",
        deps=cast(Deps, cast(Any, object())),
        model=model,
    )

    parameters = model.last_model_request_parameters
    assert parameters is not None
    assert [tool.name for tool in parameters.function_tools] == [
        "browse_tables",
        "find_columns",
        "describe_table",
        "query_data",
        "inspect_table_output",
        "display_table_output",
    ]
