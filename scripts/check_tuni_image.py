"""One opt-in TUNI image-input probe using a synthetic chart, no benchmark data."""

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

import vl_convert as vlc
from pydantic_ai import BinaryContent
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, TextPart, ThinkingPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModelSettings

from src.agent.config import AgentConfig

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "tuni-image-probe"
PROMPT = (
    'Read the attached chart. Reply only with a JSON object containing "title", '
    '"tallest_bar_label", "tallest_bar_value", and "bars_count". '
    "Read these from the image; do not guess."
)
EXPECTED = {
    "title": "VISION 7P9K",
    "tallest_bar_label": "Birch",
    "tallest_bar_value": 19,
    "bars_count": 3,
}


def prepare_image() -> Path:
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": 500,
        "height": 280,
        "title": {"text": EXPECTED["title"], "fontSize": 24},
        "data": {"values": [
            {"label": "Cedar", "value": 13},
            {"label": "Oriel", "value": 7},
            {"label": "Birch", "value": 19},
        ]},
        "encoding": {
            "x": {"field": "label", "type": "nominal", "sort": None,
                  "axis": {"labelAngle": 0, "labelFontSize": 18}, "title": None},
            "y": {"field": "value", "type": "quantitative",
                  "scale": {"domain": [0, 22]}, "title": "Value"},
        },
        "layer": [
            {"mark": {"type": "bar", "color": "#3477bc"}},
            {"mark": {"type": "text", "dy": -12, "fontSize": 18},
             "encoding": {"text": {"field": "value", "type": "quantitative"}}},
        ],
        "config": {"background": "white"},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path = OUTPUT_DIR / "synthetic-chart.png"
    image_path.write_bytes(vlc.vegalite_to_png(spec, scale=2))
    return image_path


async def probe(image_path: Path) -> None:
    config = AgentConfig(
        model_provider="tuni",
        model_name="Qwen3.8-27B-AIC",
        model_request_max_attempts=1,
        model_request_retry_budget_seconds=60,
    )
    model = config.create_model()
    model.client.max_retries = 0
    image_bytes = image_path.read_bytes()
    record = {
        "checked_at": datetime.now(UTC).isoformat(),
        "model": model.model_name,
        "provider_url": config.load_provider().url,
        "pydantic_model": type(model).__name__,
        "image_path": str(image_path),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "image_bytes": len(image_bytes),
        "prompt": PROMPT,
        "expected": EXPECTED,
        "max_tokens": 1024,
        "timeout_seconds": 60,
        "automatic_retries": 0,
    }
    started = monotonic()
    print("Sending one synthetic PNG to the configured TUNI Qwen endpoint.", flush=True)
    try:
        response = await model.request(
            [ModelRequest(parts=[UserPromptPart([
                PROMPT,
                BinaryContent(data=image_bytes, media_type="image/png"),
            ])])],
            OpenAIChatModelSettings(max_tokens=1024, timeout=60),
            ModelRequestParameters(),
        )
        final_text = "".join(
            part.content for part in response.parts if isinstance(part, TextPart)
        ).strip()
        try:
            parsed = json.loads(final_text)
        except json.JSONDecodeError:
            parsed = None
        record.update(
            accepted=True,
            finish_reason=response.finish_reason,
            final_answer=final_text,
            parsed_answer=parsed,
            correct=parsed == EXPECTED,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            thinking_parts=sum(
                isinstance(part, ThinkingPart) for part in response.parts
            ),
        )
    except (ModelAPIError, UnexpectedModelBehavior, TimeoutError) as error:
        record.update(
            accepted=False,
            error_type=type(error).__name__,
            status_code=getattr(error, "status_code", None),
        )
    finally:
        record["elapsed_seconds"] = round(monotonic() - started, 2)
        await model.client.close()
        (OUTPUT_DIR / "result.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n"
        )
    print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)
    if not record.get("correct"):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    image = prepare_image()
    if args.prepare_only:
        print(image)
    else:
        asyncio.run(probe(image))
