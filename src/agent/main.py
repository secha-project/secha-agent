import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai import ModelSettings, UsageLimits

from .agent import create_agent
from .artifacts.base import Artifact, ArtifactContext
from .catalog import UnityCatalogClient
from .config import AgentConfig, create_model
from .deps import Deps
from .semantic import load_semantic_runtime
from .spark import get_spark_session


@dataclass(frozen=True)
class AgentOutcome:
    output: str
    artifacts: tuple[Artifact, ...]


def run(prompt: str, config: AgentConfig | None = None) -> AgentOutcome:
    config = config or AgentConfig()  # type: ignore[call-arg]
    spark = get_spark_session(config)
    semantic = (
        load_semantic_runtime(spark)
        if "semantic" in config.enabled_capabilities
        else None
    )
    artifact_context = ArtifactContext()
    deps = Deps(
        spark=spark,
        catalog=UnityCatalogClient(
            config.catalog_url,
            config.catalog_token,
            config.catalog_name,
        ),
        schema_name=config.schema_name,
        table_column_page_size=config.table_column_page_size,
        semantic=semantic,
        artifact_context=artifact_context,
    )
    agent = create_agent(config.enabled_capabilities)
    result = agent.run_sync(
        prompt,
        deps=deps,
        model=create_model(config),
        model_settings=ModelSettings(temperature=config.temperature),
        usage_limits=UsageLimits(request_limit=config.request_limit),
    )
    return AgentOutcome(
        output=result.output,
        artifacts=artifact_context.published_artifacts,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the SECHA agent once.")
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Question for the agent. Reads stdin when omitted.",
    )
    args = parser.parse_args(argv)
    prompt = sys.stdin.read().strip() if args.prompt is None else args.prompt.strip()
    if not prompt:
        parser.error("provide a prompt argument or pipe a prompt on stdin")

    print(run(prompt).output)


if __name__ == "__main__":
    main()
