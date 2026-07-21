from pathlib import Path
from typing import Literal

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    catalog_url: str
    catalog_token: str
    catalog_name: str = "analytics"
    spark_url: str
    schema_name: str = "bronze"
    table_column_page_size: int = 20
    model_provider: Literal["tuni", "openai"] = "tuni"
    tuni_inference_url: str
    tuni_inference_api_key: str
    tuni_inference_model_name: str = "granite4-32b"
    openai_api_key: str | None = None
    openai_model_name: str = "gpt-5.4-mini"
    otel_exporter_otlp_endpoint: str = "http://localhost:6006"
    logfire_service_name: str = "secha-agent"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env"
    )


def get_model(app_config: AppConfig) -> OpenAIChatModel:
    if app_config.model_provider == "tuni":
        model_name = app_config.tuni_inference_model_name
        provider = OpenAIProvider(
            base_url=app_config.tuni_inference_url,
            api_key=app_config.tuni_inference_api_key,
        )
    elif app_config.model_provider == "openai":
        model_name = app_config.openai_model_name
        provider = OpenAIProvider(api_key=app_config.openai_api_key)
    else:
        raise ValueError(
            f"Unknown model provider {app_config.model_provider!r}; "
            "use 'tuni' or 'openai'."
        )

    return OpenAIChatModel(model_name, provider=provider)
