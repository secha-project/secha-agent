from collections.abc import Callable, Iterable
from functools import cache
from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined
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
PROMPT_ENVIRONMENT = Environment(
    loader=FileSystemLoader(PROMPT_ROOT),
    autoescape=False,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


@cache
def render_prompt(
    relative_path: str,
    capabilities: SelectedCapabilities,
) -> str:
    prompt = PROMPT_ENVIRONMENT.get_template(relative_path).render(
        capabilities=capabilities,
    ).strip()
    if not prompt:
        raise ValueError(f"Rendered prompt is empty: {relative_path}")
    return prompt


def build_catalog(selected: SelectedCapabilities) -> Capability[Deps]:
    return Capability(
        id="catalog",
        instructions=render_prompt(
            "capabilities/catalog/instructions.md",
            selected,
        ),
        toolsets=[catalog_toolset],
    )


def build_query(selected: SelectedCapabilities) -> Capability[Deps]:
    return Capability(
        id="query",
        instructions=render_prompt(
            "capabilities/query/instructions.md",
            selected,
        ),
        toolsets=[query_toolset],
    )


def build_semantic(selected: SelectedCapabilities) -> Capability[Deps]:
    return Capability(
        id="semantic",
        instructions=render_prompt(
            "capabilities/semantic/instructions.md",
            selected,
        ),
        toolsets=[semantic_toolset],
    )


def build_visualization(selected: SelectedCapabilities) -> Capability[Deps]:
    return Capability(
        id="visualization",
        instructions=render_prompt(
            "capabilities/visualization/instructions.md",
            selected,
        ),
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
        instructions=render_prompt("base.md", selected),
        capabilities=capabilities,
    )
