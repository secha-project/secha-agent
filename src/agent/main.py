import os

import logfire

from .catalog import UnityCatalogClient
from .config import AppConfig, get_model
from .spark_utils import get_spark_session
from .agent import Deps, agent


def main() -> None:
    config = AppConfig()  # type: ignore[call-arg]
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = config.otel_exporter_otlp_endpoint
    logfire.configure(
        send_to_logfire=False,
        service_name=config.logfire_service_name,
        metrics=False,
    )
    logfire.instrument_pydantic_ai()

    deps = Deps(
        spark=get_spark_session(config),
        catalog=UnityCatalogClient(
            config.catalog_url,
            config.catalog_token,
            config.catalog_name,
        ),
        schema_name=config.schema_name,
        table_column_page_size=config.table_column_page_size,
    )
    model = get_model(config)

    user_prompt = """What are the units of the import active energy counter column 
    in power_quality_mesurements_electrix?"""

    result = agent.run_sync(
        user_prompt,
        deps=deps,
        model=model,
    )
    print(result.output)


if __name__ == "__main__":
    main()
