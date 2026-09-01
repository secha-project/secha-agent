import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai import ModelSettings, UsageLimits

from .agent import create_agent
from .catalog import UnityCatalogClient
from .config import AgentConfig
from .deps import Deps
from .outputs.artifacts import Artifact
from .spark_runtime import get_spark_session


@dataclass(frozen=True)
class AgentOutcome:
    output: str
    artifacts: tuple[Artifact, ...]


def run(prompt: str, config: AgentConfig | None = None) -> AgentOutcome:
    if config is None:
        config = AgentConfig()  # type: ignore[call-arg]
    spark = get_spark_session(config)
    deps = Deps(
        spark=spark,
        catalog=UnityCatalogClient(
            config.catalog_url,
            config.catalog_token,
            config.catalog_name,
        ),
        schema_name=config.schema_name,
        table_column_page_size=config.table_column_page_size,
        semantic_layer_enabled="semantic_layer" in config.enabled_capabilities,
    )
    agent = create_agent(config.enabled_capabilities)
    result = agent.run_sync(
        prompt,
        deps=deps,
        model=config.create_model(),
        model_settings=ModelSettings(temperature=config.temperature),
        usage_limits=UsageLimits(request_limit=config.request_limit),
    )
    return AgentOutcome(
        output=result.output,
        artifacts=deps.artifacts.published,
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
