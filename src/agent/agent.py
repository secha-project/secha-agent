from collections.abc import Iterable, Mapping

from pydantic_ai import Agent

from .capabilities import CAPABILITIES, validate
from .deps import Deps
from .prompt import render_prompt


def create_agent(
    enabled_capabilities: Iterable[str],
    *,
    prompt_overrides: Mapping[str, str] | None = None,
) -> Agent[Deps, str]:
    """Build an agent with optional final prompt text keyed by template path."""
    selected = validate(enabled_capabilities)
    overrides = dict(prompt_overrides) if prompt_overrides is not None else None
    if overrides is not None and any(
        not isinstance(value, str) or not value.strip() for value in overrides.values()
    ):
        raise ValueError("Prompt overrides must contain nonempty strings")

    return Agent(
        deps_type=Deps,
        name="secha_agent",
        instructions=render_prompt("base.md", selected, overrides=overrides),
        capabilities=[
            capability.build(selected, prompt_overrides=overrides)
            for capability in CAPABILITIES
            if capability.NAME in selected
        ],
    )
