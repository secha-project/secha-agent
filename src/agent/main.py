import os
from uuid import uuid4

import logfire
from pydantic_ai import ModelSettings, UsageLimits

from src.app.dataframes import DataFrameReferences
from src.app.storage import ChatRepository

from .agent import agent
from .catalog import UnityCatalogClient
from .config import AgentConfig
from .deps import Deps
from .providers import get_model
from .spark import get_spark_session


def main() -> None:
    config = AgentConfig(  # type: ignore[call-arg]
        model_provider="openai",
        openai={"model_name": "gpt-5.4-mini"},
    )
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = config.otel_exporter_otlp_endpoint
    logfire.configure(
        send_to_logfire=False,
        service_name=config.logfire_service_name,
        metrics=False,
    )
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)

    spark = get_spark_session(config)
    repository = ChatRepository()
    repository.initialize()
    user = repository.create_user(f"cli-{uuid4().hex}", uuid4().hex)
    conversation = repository.create_conversation(user.id)
    deps = Deps(
        spark=spark,
        catalog=UnityCatalogClient(
            config.catalog_url,
            config.catalog_token,
            config.catalog_name,
        ),
        schema_name=config.schema_name,
        table_column_page_size=config.table_column_page_size,
        dataframe_references=DataFrameReferences(
            spark,
            repository,
            conversation.id,
        ),
    )
    model = get_model(config)

    user_prompt = """What are the units of the import active energy counter column 
    in power_quality_mesurements_electrix?"""

    result = agent.run_sync(
        user_prompt,
        deps=deps,
        model=model,
        model_settings=ModelSettings(temperature=config.temperature),
        usage_limits=UsageLimits(request_limit=config.request_limit),
    )
    print(result.output)


if __name__ == "__main__":
    main()
