from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_settings import BaseSettings, SettingsConfigDict

from .agent import CapabilityName


@dataclass(frozen=True)
class TuniInferenceConfig:
    url: str
    api_key: str = field(repr=False)
    model_name: str = "granite4-32b"


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str = field(repr=False)
    model_name: str = "gpt-5.4-mini"


ModelProviderConfig = TuniInferenceConfig | OpenAIConfig


class AgentConfig(BaseSettings):
    catalog_url: str
    catalog_token: str
    catalog_name: str = "analytics"
    spark_url: str
    schema_name: str = "bronze"
    table_column_page_size: int = 20
    model_provider: Literal["tuni", "openai"]
    tuni_inference: TuniInferenceConfig | None = None
    openai: OpenAIConfig | None = None
    temperature: float = 0.0
    request_limit: int = 50
    enabled_capabilities: tuple[CapabilityName, ...]
    otel_exporter_otlp_endpoint: str
    logfire_service_name: str = "secha-agent"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_nested_delimiter="_",
        env_nested_max_split=1,
    )

    def get_provider_config(self) -> ModelProviderConfig:
        """Return the selected provider config, failing clearly if it is missing."""
        if self.model_provider == "tuni":
            if self.tuni_inference is None:
                raise ValueError(
                    "TUNI_INFERENCE_URL and TUNI_INFERENCE_API_KEY are required "
                    "when model_provider='tuni'."
                )
            return self.tuni_inference

        if self.openai is None:
            raise ValueError("OPENAI_API_KEY is required when model_provider='openai'.")
        return self.openai

    def experiment_metadata(self) -> dict[str, Any]:
        """Return task-relevant metadata without credentials or inactive providers."""
        provider_settings = asdict(self.get_provider_config())
        provider_settings.pop("api_key")

        return {
            "model_provider": self.model_provider,
            "provider_settings": provider_settings,
            "temperature": self.temperature,
            "request_limit": self.request_limit,
            "enabled_capabilities": list(self.enabled_capabilities),
            "catalog_name": self.catalog_name,
            "schema_name": self.schema_name,
            "table_column_page_size": self.table_column_page_size,
        }


def create_model(app_config: AgentConfig) -> OpenAIChatModel:
    provider_config = app_config.get_provider_config()

    if isinstance(provider_config, TuniInferenceConfig):
        provider = OpenAIProvider(
            base_url=provider_config.url,
            api_key=provider_config.api_key,
        )
    else:
        provider = OpenAIProvider(api_key=provider_config.api_key)

    return OpenAIChatModel(provider_config.model_name, provider=provider)
