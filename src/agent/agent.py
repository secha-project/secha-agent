from collections.abc import Iterable, Mapping

from pydantic_ai import Agent

from .capabilities import CAPABILITIES, validate
from .deps import Deps
from .prompt import prompt_files, render_prompt


def create_agent(
    enabled_capabilities: Iterable[str],
    *,
    prompt_overrides: Mapping[str, str] | None = None,
) -> Agent[Deps, str]:
    selected = validate(enabled_capabilities)
    overrides = dict(prompt_overrides) if prompt_overrides is not None else None
    if overrides is not None:
        unknown = overrides.keys() - prompt_files(selected).keys()
        if unknown:
            raise ValueError(
                f"Unknown or inactive prompt components: {sorted(unknown)}"
            )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in overrides.values()
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
