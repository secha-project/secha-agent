from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.models.zai import ZaiModel
from pydantic_ai.profiles import InlineDefsJsonSchemaTransformer, ModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.zai import ZaiProvider
from pydantic_settings import BaseSettings, SettingsConfigDict

from .capabilities import EnabledCapabilities

DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 600

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
DEFAULT_ENABLED_CAPABILITIES = frozenset(
    {"catalog", "query", "semantic_layer", "table_outputs", "visualization"}
)


class PricingReference(BaseModel):
    """A model reference for estimating custom-provider costs."""

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


class ZaiProviderConfig(BaseSettings):
    api_key: SecretStr
    model_name: str = "glm-5.3-flash"
    url: str = "https://api.z.ai/api/paas/v4/"

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_prefix="ZAI_",
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
    model_provider: Literal["tuni", "openai", "zai"] = "tuni"
    model_name: str | None = None
    pricing: PricingReference | None = None
    temperature: float | None = Field(default=None, ge=0.0)
    request_limit: int = Field(default=50, ge=1)
    model_request_max_attempts: int = Field(default=6, ge=1)
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

    def load_provider(
        self,
    ) -> TuniProviderConfig | OpenAIProviderConfig | ZaiProviderConfig:
        if self.model_provider == "tuni":
            return TuniProviderConfig()
        if self.model_provider == "openai":
            return OpenAIProviderConfig()
        return ZaiProviderConfig()

    def create_model(self) -> OpenAIChatModel | OpenAIResponsesModel:
        provider_config = self.load_provider()
        api_key = provider_config.api_key.get_secret_value()
        model_name = self.model_name or provider_config.model_name
        profile = None

        if isinstance(provider_config, TuniProviderConfig) and (
            model_name.lower().startswith("qwen")
        ):
            # TUNI aliases need Qwen's streamed leading-whitespace handling.
            profile = qwen_model_profile(model_name.lower())

        provider = create_provider(
            zai=isinstance(provider_config, ZaiProviderConfig),
            base_url=(
                provider_config.url
                if isinstance(provider_config, (TuniProviderConfig, ZaiProviderConfig))
                else None
            ),
            api_key=api_key,
            max_attempts=self.model_request_max_attempts,
        )

        if isinstance(provider_config, ZaiProviderConfig):
            # Inline nested schemas after GLM emitted JSON strings for refs.
            # Keep Z.ai's native reasoning defaults and argument validation.
            return ZaiModel(
                model_name,
                provider=provider,
                profile=ModelProfile(
                    json_schema_transformer=InlineDefsJsonSchemaTransformer
                ),
            )

        model_class = (
            OpenAIResponsesModel
            if isinstance(provider_config, OpenAIProviderConfig)
            and model_name.startswith("gpt-6-")
            else OpenAIChatModel
        )
        return model_class(
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
        if isinstance(provider_config, ZaiProviderConfig):
            # Prevent resuming an old run with a different tool schema/adapter.
            provider_settings["model_adapter"] = "zai_inline_defs_v1"

        return {
            "model_provider": self.model_provider,
            "provider_settings": provider_settings,
            "pricing": self.pricing.model_dump(mode="json") if self.pricing else None,
            "temperature": self.temperature,
            "request_limit": self.request_limit,
            "model_request_max_attempts": self.model_request_max_attempts,
            "enabled_capabilities": sorted(self.enabled_capabilities),
            "catalog_name": self.catalog_name,
            "schema_name": self.schema_name,
            "table_column_page_size": self.table_column_page_size,
        }


async def _provider_retry_hint(response: httpx.Response) -> None:
    """Override the SDK's retry decision only for TUNI and exhausted quota."""
    if response.status_code not in {400, 429}:
        return
    await response.aread()
    try:
        body = response.json()
    except ValueError:
        return
    # AsyncOpenAI honors this response header before its normal status checks.
    match response.status_code, body:
        case 400, {"detail": "Open WebUI: Server Connection Error"}:
            response.headers["x-should-retry"] = "true"
        case 429, {
            "error": {"code": "insufficient_quota" | "billing_hard_limit_reached"}
        }:
            response.headers["x-should-retry"] = "false"


def create_provider(
    *,
    api_key: str,
    base_url: str | None = None,
    zai: bool = False,
    max_attempts: int = 6,
) -> OpenAIProvider | ZaiProvider:
    provider = (
        ZaiProvider(api_key=api_key)
        if zai
        else OpenAIProvider(api_key=api_key, base_url=base_url)
    )
    if zai and base_url:
        provider.client.base_url = base_url
    provider.client.max_retries = max_attempts - 1
    http_client = provider.client._client
    http_client.event_hooks["response"].append(_provider_retry_hint)
    # Preserve the hook when Pydantic AI recreates a closed, owned client.
    provider._http_client_factory = partial(
        httpx.AsyncClient,
        timeout=http_client.timeout,
        headers=http_client.headers,
        event_hooks=http_client.event_hooks,
    )
    return provider
