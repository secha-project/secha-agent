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

# Stable component names used in experiments; paths remain the file defaults.
PROMPT_FILES = {
    "base.instructions": "base.md",
    "few_shot.instructions": "capabilities/few_shot/instructions.md",
    "semantic_layer.instructions": "capabilities/semantic_layer/instructions.md",
    "catalog.instructions": "capabilities/catalog/instructions.md",
    "query.instructions": "capabilities/query/instructions.md",
    "table_outputs.instructions": "capabilities/table_outputs/instructions.md",
    "visualization.instructions": "capabilities/visualization/instructions.md",
    "browse_tables.description": "capabilities/catalog/tools/browse_tables.md",
    "find_columns.description": "capabilities/catalog/tools/find_columns.md",
    "describe_table.description": "capabilities/catalog/tools/describe_table.md",
    "query_data.description": "capabilities/query/tools/query_data.md",
    "inspect_table_output.description": "capabilities/table_outputs/tools/inspect_table_output.md",
    "display_table_output.description": "capabilities/table_outputs/tools/display_table_output.md",
    "visualize_table_output.description": "capabilities/visualization/tools/visualize_table_output.md",
}
_PROMPT_KEYS = {path: key for key, path in PROMPT_FILES.items()}


def prompt_files(capabilities: frozenset[str]) -> dict[str, str]:
    """Return named prompt files used by this capability selection."""
    return {
        key: path
        for key, path in PROMPT_FILES.items()
        if path == "base.md" or path.split("/")[1] in capabilities
    }


def snapshot_prompts(capabilities: frozenset[str]) -> dict[str, str]:
    """Freeze the rendered text for the selected benchmark configuration."""
    return {
        key: render_prompt(path, capabilities)
        for key, path in prompt_files(capabilities).items()
    }


def render_prompt(
    relative_path: str,
    capabilities: frozenset[str],
    *,
    overrides: Mapping[str, str] | None = None,
) -> str:
    key = _PROMPT_KEYS.get(relative_path)
    if overrides is not None and key is not None and key in overrides:
        # Overrides are final model-facing text; Jinja belongs to file defaults.
        prompt = overrides[key].strip()
        if not prompt:
            raise ValueError(f"Rendered prompt is empty: {key}")
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
