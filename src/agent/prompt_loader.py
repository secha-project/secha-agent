from functools import cache
from pathlib import Path

PROMPT_ROOT = Path(__file__).with_name("prompts")


@cache
def load_prompt(relative_path: str) -> str:
    prompt = (PROMPT_ROOT / relative_path).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt is empty: {relative_path}")
    return prompt


def compose_prompts(*relative_paths: str) -> str:
    return "\n\n".join(load_prompt(path) for path in relative_paths)
