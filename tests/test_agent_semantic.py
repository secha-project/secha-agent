import pytest

from src.agent.agent import create_agent


def test_semantic_capability_requires_native_catalog_query_stack() -> None:
    with pytest.raises(
        ValueError,
        match="semantic_layer capability requires catalog, query, and table_outputs",
    ):
        create_agent(("semantic_layer",))
