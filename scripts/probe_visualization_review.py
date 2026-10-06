"""Render a synthetic chart; opt in to a capped live image-review experiment."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from typing import Any, Literal, cast
from uuid import uuid4

import pandas as pd
from pydantic_ai import (
    AgentRunResultEvent,
    BinaryContent,
    FunctionToolCallEvent,
    ModelSettings,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UsageLimits,
    UserPromptPart,
)
from pydantic_ai.exceptions import AgentRunError

from src.agent.agent import create_agent
from src.agent.capabilities.visualization import _render_preview, visualize_table_output
from src.agent.config import AgentConfig
from src.agent.deps import Deps
from src.agent.model_retry import (
    DEFAULT_MODEL_REQUEST_RETRY_BUDGET_SECONDS,
    DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS,
)
from src.agent.outputs.artifacts import ChartArtifact
from src.agent.outputs.charts import ChartOutput
from src.agent.outputs.tables import LiveTableOutput


def bar_values(spec: dict[str, Any]) -> tuple[tuple[str, float], ...] | None:
    """Compare category/value pairs independently of ggsql's axis projection."""
    layers = spec.get("layer", [spec])
    if len(layers) != 1:
        return None
    layer = layers[0]
    mark = layer.get("mark")
    if (mark.get("type") if isinstance(mark, dict) else mark) != "bar":
        return None
    axes = [layer.get("encoding", {}).get(axis, {}) for axis in ("x", "y")]
    category = next(
        (axis.get("field") for axis in axes if axis.get("type") == "nominal"), None
    )
    value = next(
        (axis.get("field") for axis in axes if axis.get("type") == "quantitative"), None
    )
    rows = spec.get("datasets", {}).get(spec.get("data", {}).get("name"))
    if not category or not value or not isinstance(rows, list):
        return None
    try:
        return tuple(sorted((str(row[category]), float(row[value])) for row in rows))
    except (KeyError, TypeError, ValueError):
        return None


def review_evidence(messages: list[Any]) -> list[dict[str, Any]]:
    """Record image handoffs and tool calls without image payloads or reasoning."""
    evidence = []
    for index, message in enumerate(messages):
        for part in message.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, list):
                for item in part.content:
                    if isinstance(item, BinaryContent):
                        evidence.append(
                            {
                                "message": index,
                                "kind": "image",
                                "sha256": hashlib.sha256(item.data).hexdigest(),
                                "bytes": len(item.data),
                            }
                        )
            elif isinstance(part, ToolCallPart):
                evidence.append(
                    {
                        "message": index,
                        "kind": "tool_call",
                        "tool": part.tool_name,
                        "args": part.args_as_dict(),
                    }
                )
            elif isinstance(part, ToolReturnPart | RetryPromptPart):
                evidence.append(
                    {
                        "message": index,
                        "kind": "retry"
                        if isinstance(part, RetryPromptPart)
                        else "tool_return",
                        "tool": part.tool_name,
                    }
                )
    return evidence


async def probe(
    output: Path,
    *,
    live: bool,
    provider: Literal["zai", "openai", "tuni"] = "zai",
    model_name: str | None = None,
    case: Literal["units", "control"] = "units",
    request_timeout_seconds: float = DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS,
    run_timeout_seconds: float = DEFAULT_MODEL_REQUEST_RETRY_BUDGET_SECONDS,
) -> dict[str, Any]:
    started = monotonic()
    frame = pd.DataFrame(
        {"site": ["North", "East", "South", "West"], "power": [12, 19, 27, 15]}
    )
    deps = Deps(
        spark=cast(Any, None),
        catalog=cast(Any, None),
        schema_name="bronze",
        table_column_page_size=20,
    )

    async def sample(_max_rows: int) -> tuple[pd.DataFrame, bool]:
        return frame.copy(), False

    table_ref = deps.table_outputs.add(
        cast(LiveTableOutput, SimpleNamespace(sample=sample))
    )
    # Neither this code nor the original axis label appears in the text prompt.
    image_code = "CHECK-" + uuid4().hex[:8].upper()
    original_axis = "Energy (kWh)" if case == "units" else "Power (kW)"
    preview = await visualize_table_output(
        cast(Any, SimpleNamespace(deps=deps)),
        table_ref,
        f"VISUALISE site AS x, power AS y DRAW bar LABEL x => 'Site', y => '{original_axis}'",
        f"Power by site | {image_code}",
    )
    output.mkdir(parents=True, exist_ok=True)
    assert isinstance(preview.return_value, ChartOutput)
    assert isinstance(preview.content, list)
    image = preview.content[1]
    assert isinstance(image, BinaryContent)
    draft = deps.chart_outputs.resolve(preview.return_value.output_ref)
    (output / "draft.json").write_bytes(draft.spec_json)
    (output / "draft.png").write_bytes(image.data)
    summary: dict[str, Any] = {
        "live": live,
        "provider": provider,
        "case": case,
        "image_only_code": image_code,
        "original_axis": original_axis,
        "draft_published": bool(deps.artifacts.published),
        "passed": False,
        "request_timeout_seconds": request_timeout_seconds,
        "run_timeout_seconds": run_timeout_seconds,
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(summary, indent=2) + "\n")
    if not live:
        return summary

    config = AgentConfig(
        model_provider=provider,
        model_name=model_name,
        model_request_max_attempts=2,
        model_request_retry_budget_seconds=run_timeout_seconds,
        enabled_capabilities=frozenset({"query", "table_outputs", "visualization"}),
    )
    agent = create_agent(config.enabled_capabilities)
    model = config.create_model()
    assert model.provider is not None
    prompt = (
        "Review this draft chart for a request to compare power by site in kW. "
        "The saved table below is correct; use it directly, without SQL or catalog calls. "
        "Inspect the supplied image, correct any concrete problems using the visualization "
        "tool, inspect the new preview, and publish only the satisfactory chart. "
        "If the draft is already correct, publish it directly without rebuilding it. "
        "Keep the draft title including its CHECK reference code. In your final answer, "
        "quote the draft's original y-axis label and its full CHECK reference code exactly "
        "as read from the image, then briefly explain any changes.\n"
        f"Table reference: {table_ref.model_dump_json()}\n"
        f"Table (power is in kW):\n{frame.to_csv(index=False)}\n"
        f"Draft chart reference: {preview.return_value.output_ref.model_dump_json()}"
    )
    calls = []
    result = None
    try:
        async with (
            asyncio.timeout(run_timeout_seconds),
            model.provider,
            agent.run_stream_events(
                [prompt, *preview.content],
                deps=deps,
                model=model,
                model_settings=ModelSettings(
                    max_tokens=8_192, timeout=request_timeout_seconds
                ),
                usage_limits=UsageLimits(request_limit=8, total_tokens_limit=40_000),
            ) as events,
        ):
            async for event in events:
                if isinstance(event, FunctionToolCallEvent):
                    calls.append(
                        {
                            "tool": event.part.tool_name,
                            "args": event.part.args_as_dict(),
                        }
                    )
                    print(
                        f"{model.model_name} {case}: {event.part.tool_name}", flush=True
                    )
                elif isinstance(event, AgentRunResultEvent):
                    result = event.result
    except (AgentRunError, TimeoutError) as error:
        api_key = config.load_provider().api_key.get_secret_value()
        summary.update(
            model=model.model_name,
            error_type=type(error).__name__,
            error=str(error).replace(api_key, "[redacted]")[:1_000],
            tool_calls=calls,
            elapsed_seconds=round(monotonic() - started, 2),
        )
        result_path.write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    if result is None:
        raise RuntimeError("The agent did not finish the review experiment.")
    charts = [
        artifact
        for artifact in deps.artifacts.published
        if isinstance(artifact, ChartArtifact)
    ]
    for index, chart in enumerate(charts, start=1):
        (output / f"published-{index}.json").write_bytes(chart.spec_json)
        (output / f"published-{index}.png").write_bytes(
            _render_preview(chart.to_spec())
        )
    evidence = review_evidence(result.all_messages())
    publications = [
        item
        for item in evidence
        if item["kind"] == "tool_call" and item["tool"] == "display_chart_output"
    ]
    images_before_publication = [
        item
        for item in evidence
        if item["kind"] == "image"
        and publications
        and item["message"] < publications[0]["message"]
    ]
    chart = charts[0] if len(charts) == 1 else None
    checks = {
        "read_image_only_code": image_code in result.output,
        "read_original_axis": original_axis in result.output,
        "one_chart_published": len(charts) == len(deps.artifacts.published) == 1,
        "data_unchanged": chart is not None
        and bar_values(chart.spec)
        == tuple(sorted(zip(frame["site"], frame["power"], strict=True))),
        "correct_axis_units": chart is not None
        and b"Power (kW)" in chart.spec_json
        and b"kWh" not in chart.spec_json,
        "title_preserved": chart is not None
        and chart.spec.get("title") == draft.spec.get("title"),
        "preview_accepted_before_publication": len(images_before_publication)
        >= (2 if case == "units" else 1),
        "appropriate_revision": chart is not None
        and (
            chart.id != draft.id
            if case == "units"
            else chart.id == draft.id
            and not any(call["tool"] == "visualize_table_output" for call in calls)
        ),
    }
    summary.update(
        model=model.model_name,
        checks=checks,
        passed=all(checks.values()),
        published_count=len(charts),
        tool_calls=calls,
        evidence=evidence,
        response=result.output,
        elapsed_seconds=round(monotonic() - started, 2),
        usage={
            "requests": result.usage.requests,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
        },
    )
    result_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Send synthetic chart images to the selected configured provider.",
    )
    parser.add_argument("--provider", choices=("zai", "openai", "tuni"), default="zai")
    parser.add_argument(
        "--model", help="Exact model ID; defaults to the provider's configured model."
    )
    parser.add_argument("--case", choices=("units", "control"), default="units")
    parser.add_argument(
        "--request-timeout", type=float, default=DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--run-timeout", type=float, default=DEFAULT_MODEL_REQUEST_RETRY_BUDGET_SECONDS
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/visualization-review-probe")
    )
    args = parser.parse_args()
    summary = asyncio.run(
        probe(
            args.output,
            live=args.live,
            provider=args.provider,
            model_name=args.model,
            case=args.case,
            request_timeout_seconds=args.request_timeout,
            run_timeout_seconds=args.run_timeout,
        )
    )
    print(json.dumps(summary, indent=2))
    if args.live and not summary["passed"]:
        raise SystemExit("The review did not pass every image and publication check.")


if __name__ == "__main__":
    main()
