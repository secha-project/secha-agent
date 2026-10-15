from collections.abc import Mapping

from src.agent.prompt import PROMPT_ROOT, render_prompt


def prompt_files(capabilities: frozenset[str]) -> dict[str, str]:
    """Discover optimization components from the agent's prompt layout."""
    files = {"base.instructions": "base.md"}
    for capability in sorted(capabilities):
        directory = PROMPT_ROOT / "capabilities" / capability
        instructions = directory / "instructions.md"
        if instructions.is_file():
            files[f"{capability}.instructions"] = instructions.relative_to(
                PROMPT_ROOT
            ).as_posix()
        for path in sorted((directory / "tools").glob("*.md")):
            key = f"{path.stem}.description"
            if key in files:
                raise ValueError(f"Duplicate prompt component: {key}")
            files[key] = path.relative_to(PROMPT_ROOT).as_posix()
    return files


def snapshot_prompts(capabilities: frozenset[str]) -> dict[str, str]:
    """Freeze rendered prompt text under stable optimization component names."""
    return {
        key: render_prompt(path, capabilities)
        for key, path in prompt_files(capabilities).items()
    }


def to_prompt_overrides(
    prompts: Mapping[str, str], capabilities: frozenset[str]
) -> dict[str, str]:
    """Translate named optimization components to the agent's template paths."""
    files = prompt_files(capabilities)
    unknown = prompts.keys() - files.keys()
    if unknown:
        raise ValueError(f"Unknown or inactive prompt components: {sorted(unknown)}")
    return {files[key]: text for key, text in prompts.items()}
