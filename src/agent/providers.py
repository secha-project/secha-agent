from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import AgentConfig, TuniInferenceConfig


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
