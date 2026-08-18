from collections.abc import Iterable

from pydantic_ai import Agent

from .capabilities import CAPABILITIES, validate
from .deps import Deps
from .prompt import render_prompt


def create_agent(
    enabled_capabilities: Iterable[str],
) -> Agent[Deps, str]:
    selected = validate(enabled_capabilities)

    return Agent(
        deps_type=Deps,
        name="secha_agent",
        instructions=render_prompt("base.md", selected),
        capabilities=[
            capability.build(selected)
            for capability in CAPABILITIES
            if capability.NAME in selected
        ],
    )
