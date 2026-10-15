from collections.abc import Mapping
from functools import cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPT_ROOT = Path(__file__).with_name("prompts")
PROMPT_ENVIRONMENT = Environment(
    loader=FileSystemLoader(PROMPT_ROOT),
    autoescape=False,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_prompt(
    relative_path: str,
    capabilities: frozenset[str],
    *,
    overrides: Mapping[str, str] | None = None,
) -> str:
    """Render a template, optionally replacing it with final text keyed by path."""
    if overrides is not None and relative_path in overrides:
        # Overrides are final model-facing text; Jinja belongs to file defaults.
        prompt = overrides[relative_path].strip()
        if not prompt:
            raise ValueError(f"Rendered prompt is empty: {relative_path}")
        return prompt
    return _render_file_prompt(relative_path, capabilities)


@cache
def _render_file_prompt(relative_path: str, capabilities: frozenset[str]) -> str:
    prompt = (
        PROMPT_ENVIRONMENT.get_template(relative_path)
        .render(
            capabilities=capabilities,
        )
        .strip()
    )
    if not prompt:
        raise ValueError(f"Rendered prompt is empty: {relative_path}")
    return prompt
