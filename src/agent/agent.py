from collections.abc import Callable, Iterable
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, Capability

from .deps import Deps
from .tools.catalog_tools import catalog_toolset
from .tools.query_tools import query_toolset
from .tools.semantic_tools import semantic_toolset
from .tools.visualization_tools import visualization_toolset

type CapabilityName = Literal["catalog", "query", "semantic", "visualization"]
type SelectedCapabilities = frozenset[CapabilityName]
type CapabilityBuilder = Callable[
    [SelectedCapabilities],
    AbstractCapability[Deps],
]

PROMPT_ROOT = Path(__file__).with_name("prompts")


@cache
def load_prompt(relative_path: str) -> str:
    prompt = (PROMPT_ROOT / relative_path).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt is empty: {relative_path}")
    return prompt


def compose_prompts(*relative_paths: str) -> str:
    return "\n\n".join(load_prompt(path) for path in relative_paths)


def build_catalog(_selected: SelectedCapabilities) -> Capability[Deps]:
    return Capability(
        id="catalog",
        instructions=load_prompt("capabilities/catalog/instructions.md"),
        toolsets=[catalog_toolset],
    )


def build_query(selected: SelectedCapabilities) -> Capability[Deps]:
    contextual_prompt = (
        "capabilities/query/with_catalog.md"
        if "catalog" in selected
        else "capabilities/query/standalone.md"
    )
    return Capability(
        id="query",
        instructions=compose_prompts(
            "capabilities/query/core.md",
            contextual_prompt,
        ),
        toolsets=[query_toolset],
    )


def build_semantic(_selected: SelectedCapabilities) -> Capability[Deps]:
    return Capability(
        id="semantic",
        instructions=load_prompt("capabilities/semantic/instructions.md"),
        toolsets=[semantic_toolset],
    )


def build_visualization(_selected: SelectedCapabilities) -> Capability[Deps]:
    return Capability(
        id="visualization",
        instructions=load_prompt("capabilities/visualization/instructions.md"),
        toolsets=[visualization_toolset],
    )


CAPABILITY_BUILDERS: dict[CapabilityName, CapabilityBuilder] = {
    "catalog": build_catalog,
    "query": build_query,
    "semantic": build_semantic,
    "visualization": build_visualization,
}


def create_agent(
    enabled_capabilities: Iterable[CapabilityName],
) -> Agent[Deps, str]:
    names = tuple(dict.fromkeys(enabled_capabilities))
    unknown = set(names) - CAPABILITY_BUILDERS.keys()
    if unknown:
        raise ValueError(f"Unknown capabilities: {sorted(unknown)}")

    selected = frozenset(names)
    if "semantic" in selected and "query" not in selected:
        raise ValueError("The semantic capability requires query.")
    if "visualization" in selected and not {"catalog", "query"} <= selected:
        raise ValueError("The visualization capability requires catalog and query.")
    capabilities = [CAPABILITY_BUILDERS[name](selected) for name in names]

    return Agent(
        deps_type=Deps,
        name="secha_agent",
        instructions=load_prompt("base.md"),
        capabilities=capabilities,
    )
