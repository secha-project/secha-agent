"""Opt-in, capped provider-default probes; run with python -m scripts.check_model_defaults.

OpenAI echoes resolved settings. TUNI exposes template validation and token counts,
but read-only Open WebUI accounts cannot see model parameter overrides. Seeded
temperature comparisons are supporting evidence, not proof of an exact default.
No credentials or generated reasoning text are written to the result files.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

import httpx

from src.agent.config import OpenAIProviderConfig, TuniProviderConfig

SUITES = (
    "openai-defaults",
    "tuni-metadata",
    "tuni-controls",
    "tuni-effort",
    "tuni-temperature",
)
PROMPT = "Reply with exactly OK."
SAMPLING_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "reasoning_effort",
    "max_tokens",
    "chat_template_kwargs",
    "enable_thinking",
}


def summarize_completion(data: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: data[key]
        for key in (
            "model",
            "object",
            "system_fingerprint",
            "usage",
            "temperature",
            "reasoning_effort",
        )
        if key in data
    }
    result["choices"] = []
    for choice in data.get("choices", []):
        message = choice.get("message", {})
        summary: dict[str, Any] = {
            "finish_reason": choice.get("finish_reason"),
            "message_keys": list(message),
        }
        for key in ("content", "reasoning_content", "reasoning"):
            value = message.get(key)
            if isinstance(value, str):
                summary[f"{key}_characters"] = len(value)
                summary[f"{key}_sha256"] = hashlib.sha256(value.encode()).hexdigest()
        result["choices"].append(summary)
    return result


async def request(
    client: httpx.AsyncClient,
    label: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    query: dict[str, str] | None = None,
    api_key: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {"case": label}
    if payload is not None:
        row["request"] = payload
    started = monotonic()
    try:
        response = (
            await client.get(url, params=query)
            if payload is None
            else await client.post(url, json=payload)
        )
        row["http_status"] = response.status_code
        data = response.json()
        if response.status_code != 200:
            row["error"] = json.dumps(data).replace(api_key, "[redacted]")
        elif label == "model_parameters":
            row["metadata"] = {
                key: data[key]
                for key in ("id", "base_model_id", "write_access")
                if key in data
            }
            row["metadata"]["sampling_params"] = {
                key: value
                for key, value in data.get("params", {}).items()
                if key in SAMPLING_KEYS
            }
            # Empty params with write_access=False mean hidden, not unset.
        elif label in ("gpt-6-sol", "gpt-6-luna"):
            row["metadata"] = {
                key: data[key]
                for key in (
                    "model",
                    "status",
                    "temperature",
                    "top_p",
                    "reasoning",
                    "usage",
                    "incomplete_details",
                )
                if key in data
            }
        else:
            row["metadata"] = summarize_completion(data)
    except (httpx.HTTPError, ValueError) as error:
        row["error_type"] = type(error).__name__
    row["elapsed_seconds"] = round(monotonic() - started, 2)
    print(json.dumps(row), flush=True)
    return row


def tuni_cases(suite: str) -> list[tuple[str, dict[str, Any]]]:
    if suite == "tuni-controls":
        return [
            ("default", {}),
            ("thinking_off", {"chat_template_kwargs": {"enable_thinking": False}}),
            ("top_level_low", {"reasoning_effort": "low"}),
            ("temperature_invalid", {"temperature": -1}),
        ]
    if suite == "tuni-effort":
        return [("effort_default", {"temperature": 0})] + [
            (
                f"effort_{effort}",
                {
                    "temperature": 0,
                    "chat_template_kwargs": {"reasoning_effort": effort},
                },
            )
            for effort in ("xhigh", "medium", "low", "invalid-probe-value")
        ]
    if suite == "tuni-temperature":
        common = {
            "chat_template_kwargs": {"enable_thinking": False},
            "seed": 271828,
            "max_tokens": 24,
            "messages": [
                {
                    "role": "user",
                    "content": "Invent a sequence of twelve unrelated lowercase English nouns. "
                    "Output only the nouns separated by spaces.",
                }
            ],
        }
        return [("temperature_default", common)] + [
            (f"temperature_{temperature}", {**common, "temperature": temperature})
            for temperature in (1.0, 0.6)
        ]
    return []


async def probe(suite: str, max_tokens: int, output: Path) -> None:
    if suite == "openai-defaults":
        cfg = OpenAIProviderConfig()
        base = "https://api.openai.com/v1"
    else:
        cfg = TuniProviderConfig()
        base = cfg.url.rstrip("/")
    api_key = cfg.api_key.get_secret_value()
    rows = []
    record = {
        "checked_at": datetime.now(UTC).isoformat(),
        "suite": suite,
        "timeout_seconds": 45,
        "automatic_retries": 0,
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=45,
        follow_redirects=False,
    ) as client:
        if suite == "tuni-metadata":
            cases = [("model_parameters", None)]
            url = base + "/v1/models/model"
        elif suite == "openai-defaults":
            cases = [
                (
                    model,
                    {
                        "model": model,
                        "input": PROMPT,
                        "max_output_tokens": max_tokens,
                        "store": False,
                    },
                )
                for model in ("gpt-6-sol", "gpt-6-luna")
            ]
            url = base + "/responses"
        else:
            cases = [
                (
                    label,
                    {
                        "model": cfg.model_name,
                        "messages": [{"role": "user", "content": PROMPT}],
                        "max_tokens": max_tokens,
                        "stream": False,
                        **extra,
                    },
                )
                for label, extra in tuni_cases(suite)
            ]
            url = base + "/chat/completions"
            for _, payload in cases:
                payload["max_tokens"] = min(payload["max_tokens"], max_tokens)
        for label, payload in cases:
            row = await request(
                client,
                label,
                url,
                payload=payload,
                query={"id": cfg.model_name} if payload is None else None,
                api_key=api_key,
            )
            rows.append(row)
            output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            if row.get("http_status") in (401, 403, 429) or "error_type" in row:
                break
    print(f"Saved {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 256:
        parser.error("--max-tokens must be between 1 and 256")
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    root = Path(__file__).resolve().parents[1]
    output = (
        args.output
        or root / "output/model-defaults-probe" / f"{args.suite}_{timestamp}.json"
    )
    asyncio.run(probe(args.suite, args.max_tokens, output))


if __name__ == "__main__":
    main()
