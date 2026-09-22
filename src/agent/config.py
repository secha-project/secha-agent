from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_settings import BaseSettings, SettingsConfigDict

from .capabilities import EnabledCapabilities

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
DEFAULT_ENABLED_CAPABILITIES = frozenset(
    {"catalog", "query", "semantic_layer", "table_outputs", "visualization"}
)


class PricingReference(BaseModel):
    """A standard model price to use when estimating custom-provider costs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = Field(min_length=1)
    provider: str | None = Field(default=None, min_length=1)


class TuniProviderConfig(BaseSettings):
    url: str
    api_key: SecretStr
    model_name: str = "Qwen3.8-27B-AIC"

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_prefix="TUNI_INFERENCE_",
        frozen=True,
        extra="ignore",
    )


class OpenAIProviderConfig(BaseSettings):
    api_key: SecretStr
    model_name: str = "gpt-5.4-mini"

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_prefix="OPENAI_",
        frozen=True,
        extra="ignore",
    )


class AgentConfig(BaseSettings):
    catalog_url: str
    catalog_token: str
    catalog_name: str = "analytics"
    spark_url: str
    schema_name: str = "bronze"
    table_column_page_size: int = Field(default=20, ge=1)
    model_provider: Literal["tuni", "openai"] = "tuni"
    model_name: str | None = None
    pricing: PricingReference | None = None
    temperature: float = Field(default=0.0, ge=0.0)
    request_limit: int = Field(default=50, ge=1)
    enabled_capabilities: EnabledCapabilities = DEFAULT_ENABLED_CAPABILITIES
    otel_exporter_otlp_endpoint: str
    logfire_service_name: str = "secha-agent"

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_nested_delimiter="_",
        env_nested_max_split=1,
        frozen=True,
        extra="ignore",
    )

    def load_provider(self) -> TuniProviderConfig | OpenAIProviderConfig:
        if self.model_provider == "tuni":
            return TuniProviderConfig()
        return OpenAIProviderConfig()

    def create_model(self) -> OpenAIChatModel:
        provider_config = self.load_provider()
        api_key = provider_config.api_key.get_secret_value()
        model_name = self.model_name or provider_config.model_name
        profile = None

        if isinstance(provider_config, TuniProviderConfig):
            if model_name.lower().startswith("qwen"):
                # TUNI aliases need Qwen's streamed leading-whitespace handling.
                profile = qwen_model_profile(model_name.lower())
            provider = OpenAIProvider(
                base_url=provider_config.url,
                api_key=api_key,
            )
        else:
            provider = OpenAIProvider(api_key=api_key)

        return OpenAIChatModel(
            model_name,
            provider=provider,
            profile=profile,
        )

    def experiment_metadata(self) -> dict[str, Any]:
        """Return task-relevant metadata without credentials or inactive providers."""
        provider_config = self.load_provider()
        provider_settings = provider_config.model_dump(
            mode="json",
            exclude={"api_key"},
        )
        provider_settings["model_name"] = self.model_name or provider_config.model_name

        return {
            "model_provider": self.model_provider,
            "provider_settings": provider_settings,
            "pricing": self.pricing.model_dump(mode="json") if self.pricing else None,
            "temperature": self.temperature,
            "request_limit": self.request_limit,
            "enabled_capabilities": sorted(self.enabled_capabilities),
            "catalog_name": self.catalog_name,
            "schema_name": self.schema_name,
            "table_column_page_size": self.table_column_page_size,
        }
