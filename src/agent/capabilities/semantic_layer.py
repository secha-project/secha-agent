from pydantic_ai.capabilities import Capability

from ..deps import Deps
from ..prompt import render_prompt

NAME = "semantic_layer"
# This prompt will only be sent when using deferred loading of capabilities.
DESCRIPTION = "Discover and prefer governed native metric views."


def build(selected: frozenset[str]) -> Capability[Deps]:
    """Add metric-view routing policy; catalog and query provide the tools."""
    return Capability(
        id=NAME,
        description=DESCRIPTION,
        instructions=render_prompt(f"capabilities/{NAME}/instructions.md", selected),
    )
