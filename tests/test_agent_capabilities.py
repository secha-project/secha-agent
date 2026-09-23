from typing import Any, cast

import pytest
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel

from src.agent.agent import create_agent
from src.agent.deps import Deps


def test_create_agent_rejects_unknown_capabilities() -> None:
    with pytest.raises(ValueError, match=r"Unknown capabilities: \['unknown'\]"):
        create_agent(cast(Any, ("catalog", "unknown")))


@pytest.mark.parametrize(
    ("capabilities", "message"),
    [
        (("visualization",), "requires table_outputs"),
        (("table_outputs", "visualization"), "requires query"),
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
    def instructions(agent: Any) -> str | None:
        with capture_run_messages() as messages:
            agent.run_sync(
                "question",
                deps=cast(Deps, cast(Any, object())),
                model=TestModel(call_tools=[], custom_output_text="done"),
            )
        return next(
            message.instructions
            for message in messages
            if isinstance(message, ModelRequest)
        )

    standalone = create_agent(("query", "table_outputs"))
    original = instructions(standalone)
    combined = create_agent(("catalog", "query", "table_outputs"))

    assert original
    assert instructions(combined) != original
    assert instructions(standalone) == original
