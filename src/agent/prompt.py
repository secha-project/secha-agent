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


@cache
def render_prompt(relative_path: str, capabilities: frozenset[str]) -> str:
    prompt = PROMPT_ENVIRONMENT.get_template(relative_path).render(
        capabilities=capabilities,
    ).strip()
    if not prompt:
        raise ValueError(f"Rendered prompt is empty: {relative_path}")
    return prompt
