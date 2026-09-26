from collections.abc import Mapping

from pydantic_ai.capabilities import Capability

from ..deps import Deps
from ..prompt import render_prompt

NAME = "few_shot"
DESCRIPTION = "Include tool-use examples in active tool descriptions."


def build(
    selected: frozenset[str],
    *,
    prompt_overrides: Mapping[str, str] | None = None,
) -> Capability[Deps]:
    """Enable prompt examples without adding tools or changing their schemas."""
    return Capability(
        id=NAME,
        description=DESCRIPTION,
        instructions=render_prompt(
            f"capabilities/{NAME}/instructions.md", selected, overrides=prompt_overrides
        ),
    )
