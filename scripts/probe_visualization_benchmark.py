"""Run one real visualization benchmark case with normal tools and evaluators."""

import argparse
import asyncio
import json
from pathlib import Path
from time import monotonic
from typing import Any, cast

from pydantic_ai import BinaryContent, ModelSettings, UsageLimits, capture_run_messages
from pydantic_ai.messages import UserPromptPart

from scripts.probe_visualization_review import review_evidence
from src.agent.agent import create_agent
from src.validation.config import load_run_config
from src.validation.eval_models import VisualizationOutput
from src.validation.progress_log import ProgressLog
from src.validation.reporting import save_report
from src.validation.runner import (
    AgentTask,
    configure_observability,
    create_dependencies,
    evaluate,
    load_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", default="task_26")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    name, saved_config = load_run_config(args.config)
    config = saved_config.model_copy(update={"repetitions": 1, "max_concurrency": 1})
    dataset = load_dataset(config.dataset)
    dataset.cases = [case for case in dataset.cases if case.name == args.task]
    if len(dataset.cases) != 1 or not any(
        isinstance(item, VisualizationOutput)
        for item in (dataset.cases[0].expected_output or [])
    ):
        parser.error("--task must select one visualization benchmark case")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifacts = output / "artifacts"
    artifacts.mkdir()
    started = monotonic()
    summary: dict[str, Any] = {
        "config": name,
        "model": config.agent.model_name,
        "task": args.task,
        "question": dataset.cases[0].inputs,
        "repetitions": 1,
        "passed": False,
    }
    evidence = []
    try:
        configure_observability(config.agent)
        task = AgentTask(
            agent=create_agent(config.agent.enabled_capabilities),
            deps=create_dependencies(config.agent),
            model=config.agent.create_model(),
            model_settings=(
                ModelSettings(temperature=config.agent.temperature)
                if config.agent.temperature is not None
                else ModelSettings()
            ),
            usage_limits=UsageLimits(request_limit=config.agent.request_limit),
        )

        async def recorded_task(prompt: str) -> str:
            with capture_run_messages() as messages:
                try:
                    async with asyncio.timeout(1_800):
                        return await task(prompt)
                finally:
                    evidence.extend(review_evidence(messages))
                    (output / "review-evidence.json").write_text(
                        json.dumps(evidence, indent=2) + "\n"
                    )
                    image_index = 0
                    for message in messages:
                        for part in message.parts:
                            if isinstance(part, UserPromptPart) and isinstance(
                                part.content, list
                            ):
                                for item in part.content:
                                    if isinstance(item, BinaryContent):
                                        image_index += 1
                                        (
                                            output / f"preview-{image_index}.png"
                                        ).write_bytes(item.data)

        print(
            f"Running {args.task} with {config.agent.model_name}: {output}", flush=True
        )
        with ProgressLog(output / "progress.log", name, output, 1):
            report = evaluate(
                dataset,
                cast(Any, recorded_task),
                name,
                config,
                artifacts,
                output / "checkpoint.json",
            )
        save_report(report, output / "report.json")
        summary["execution_failures"] = [
            failure.error_message for failure in report.failures
        ]
        summary["report_evaluator_failures"] = [
            failure.error_message for failure in report.report_evaluator_failures
        ]
        if report.cases:
            case = report.cases[0]
            summary.update(
                assertions={key: value.value for key, value in case.assertions.items()},
                reasons={key: value.reason for key, value in case.assertions.items()},
                evaluator_failures=[
                    failure.error_message for failure in case.evaluator_failures
                ],
                metrics=case.metrics,
                response=case.output,
                task_duration_seconds=case.task_duration,
                artifacts=case.attributes.get("artifact_paths", []),
            )
            publications = [
                event
                for event in evidence
                if event.get("tool") == "display_chart_output"
                and event["kind"] == "tool_call"
            ]
            images = [event for event in evidence if event["kind"] == "image"]
            summary["image_review_verified"] = bool(publications) and all(
                any(image["message"] < publication["message"] for image in images)
                for publication in publications
            )
            summary["preview_count"] = len(images)
            summary["publication_count"] = len(publications)
            summary["passed"] = (
                bool(case.assertions)
                and all(value.value is True for value in case.assertions.values())
                and not case.evaluator_failures
                and not report.report_evaluator_failures
                and summary["image_review_verified"]
            )
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary["elapsed_seconds"] = round(monotonic() - started, 2)
        (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
