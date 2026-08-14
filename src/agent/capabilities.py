from collections.abc import Callable

from pydantic_ai.capabilities import AbstractCapability, Capability

from .deps import Deps
from .profiles import DEFAULT_CAPABILITIES, CapabilityName
from .prompt_loader import compose_prompts, load_prompt
from .tools.catalog_tools import catalog_toolset
from .tools.query_tools import query_toolset
from .tools.visualization_tools import visualization_toolset

type SelectedCapabilities = frozenset[CapabilityName]
type CapabilityBuilder = Callable[
    [SelectedCapabilities],
    AbstractCapability[Deps],
]


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


def build_visualization(_selected: SelectedCapabilities) -> Capability[Deps]:
    return Capability(
        id="visualization",
        instructions=load_prompt(
            "capabilities/visualization/instructions.md"
        ),
        toolsets=[visualization_toolset],
    )


CAPABILITY_BUILDERS: dict[CapabilityName, CapabilityBuilder] = {
    "catalog": build_catalog,
    "query": build_query,
    "visualization": build_visualization,
}
