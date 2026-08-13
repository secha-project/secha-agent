from collections.abc import Iterable

from pydantic_ai import Agent

from .capabilities import (
    CAPABILITY_BUILDERS,
    DEFAULT_CAPABILITIES,
    CapabilityName,
)
from .deps import Deps
from .prompt_loader import load_prompt


def create_agent(
    enabled_capabilities: Iterable[CapabilityName] = DEFAULT_CAPABILITIES,
) -> Agent[Deps, str]:
    names = tuple(dict.fromkeys(enabled_capabilities))
    unknown = set(names) - CAPABILITY_BUILDERS.keys()
    if unknown:
        raise ValueError(f"Unknown capabilities: {sorted(unknown)}")

    selected = frozenset(names)
    capabilities = [CAPABILITY_BUILDERS[name](selected) for name in names]

    return Agent(
        deps_type=Deps,
        name="secha_agent",
        instructions=load_prompt("base.md"),
        capabilities=capabilities,
    )


# Default application profile, built once when this module is imported.
agent = create_agent()
