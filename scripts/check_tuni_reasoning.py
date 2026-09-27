"""Three capped live Qwen probes: default, explicitly enabled, and disabled thinking."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, TextPart, ThinkingPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModelSettings

from src.validation.config import PROJECT_ROOT, load_run_config

PROMPT = "What is 17 * 23 + 19 * 7? Reply with only the final integer."
MODES = {"default": None, "enabled": True, "disabled": False}


async def probe(output: Path, modes: list[str], max_tokens: int) -> None:
    _, config = load_run_config("qwen3.8-27b__baseline")
    model = config.agent.create_model()
    model.client.max_retries = 0
    rows = []
    record = {
        "checked_at": datetime.now(UTC).isoformat(),
        "model": model.model_name,
        "pydantic_model": type(model).__name__,
        "prompt": PROMPT,
        "max_tokens_per_request": max_tokens,
        "timeout_seconds": 60,
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for mode in modes:
            settings = OpenAIChatModelSettings(max_tokens=max_tokens, timeout=60)
            enabled = MODES[mode]
            if enabled is not None:
                settings["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": enabled}
                }
            row = {"mode": mode, "settings": settings}
            started = monotonic()
            print(f"Probing TUNI Qwen: {mode}", flush=True)
            try:
                response = await model.request(
                    [ModelRequest(parts=[UserPromptPart(PROMPT)])],
                    settings,
                    ModelRequestParameters(),
                )
                thinking = [
                    part for part in response.parts if isinstance(part, ThinkingPart)
                ]
                final_text = "".join(
                    part.content
                    for part in response.parts
                    if isinstance(part, TextPart)
                ).strip()
                row.update(
                    accepted=True,
                    finish_reason=response.finish_reason,
                    thinking_parts=len(thinking),
                    thinking_characters=sum(len(part.content) for part in thinking),
                    correct_final_answer=final_text == "524",
                    final_answer_characters=len(final_text),
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    usage_details=response.usage.details,
                )
            except (ModelAPIError, UnexpectedModelBehavior, TimeoutError) as error:
                # Keep errors useful without recording credentials or response text.
                row.update(
                    accepted=False,
                    error_type=type(error).__name__,
                    status_code=getattr(error, "status_code", None),
                )
            row["elapsed_seconds"] = round(monotonic() - started, 2)
            rows.append(row)
            output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(row), flush=True)
            if not row["accepted"] and row.get("status_code") not in (400, 422):
                break
    finally:
        await model.client.close()
    print(f"Saved metadata (no reasoning text): {output}", flush=True)
    if not rows or not any(row["accepted"] for row in rows):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 2048:
        parser.error("--max-tokens must be between 1 and 2048")
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    output = args.output or (
        PROJECT_ROOT / "output" / "reasoning-probe" / f"tuni_{timestamp}.json"
    )
    asyncio.run(probe(output, args.modes, args.max_tokens))


if __name__ == "__main__":
    main()
