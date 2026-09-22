from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

import src.agent.main as main_module
from src.agent.config import (
    DEFAULT_ENABLED_CAPABILITIES,
    AgentConfig,
    OpenAIProviderConfig,
    PricingReference,
    TuniProviderConfig,
)
from src.agent.main import AgentOutcome
from src.agent.outputs.artifacts import TableArtifact


def test_agent_config_uses_default_capabilities() -> None:
    config = AgentConfig(
        _env_file=None,
        catalog_url="https://catalog.example",
        catalog_token="token",
        spark_url="sc://spark.example",
        otel_exporter_otlp_endpoint="http://otel.example",
    )

    assert config.enabled_capabilities == DEFAULT_ENABLED_CAPABILITIES


def test_agent_config_rejects_invalid_cross_field_state() -> None:
    with pytest.raises(
        ValidationError,
        match="semantic_layer capability requires catalog, query, and table_outputs",
    ):
        AgentConfig(
            _env_file=None,
            catalog_url="https://catalog.example",
            catalog_token="token",
            spark_url="sc://spark.example",
            model_provider="openai",
            enabled_capabilities=("semantic_layer",),
            otel_exporter_otlp_endpoint="http://otel.example",
        )


def test_agent_config_loads_only_the_selected_provider_from_environment(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("TUNI_INFERENCE_URL", "https://tuni.example")
    monkeypatch.setenv("TUNI_INFERENCE_API_KEY", "secret")
    monkeypatch.setenv("TUNI_INFERENCE_MODEL_NAME", "custom-model")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-openai-secret")

    config = AgentConfig(
        _env_file=None,
        catalog_url="https://catalog.example",
        catalog_token="token",
        spark_url="sc://spark.example",
        model_provider="tuni",
        enabled_capabilities=("query", "table_outputs"),
        otel_exporter_otlp_endpoint="http://otel.example",
    )
    provider = config.load_provider()

    assert isinstance(provider, TuniProviderConfig)
    assert provider.url == "https://tuni.example"
    assert provider.model_name == "custom-model"
    assert provider.api_key.get_secret_value() == "secret"
    assert "secret" not in repr(provider)
    assert "unused-openai-secret" not in repr(config)

    qwen_model = config.model_copy(
        update={"model_name": "Qwen3.8-27B-AIC"}
    ).create_model()
    assert qwen_model.model_name == "Qwen3.8-27B-AIC"
    assert str(qwen_model.client.base_url) == "https://tuni.example"
    assert qwen_model.profile["ignore_streamed_leading_whitespace"] is True
    assert not config.create_model().profile.get("ignore_streamed_leading_whitespace")

    priced_config = config.model_copy(
        update={
            "pricing": PricingReference(
                provider="openai",
                model_name="gpt-5-mini",
            )
        }
    )
    assert priced_config.experiment_metadata()["pricing"] == {
        "provider": "openai",
        "model_name": "gpt-5-mini",
    }

    openai_config = AgentConfig(
        _env_file=None,
        catalog_url="https://catalog.example",
        catalog_token="token",
        spark_url="sc://spark.example",
        model_provider="openai",
        enabled_capabilities=("query", "table_outputs"),
        otel_exporter_otlp_endpoint="http://otel.example",
    )
    openai_provider = openai_config.load_provider()

    assert isinstance(openai_provider, OpenAIProviderConfig)
    assert openai_provider.api_key.get_secret_value() == "unused-openai-secret"
    assert "https://tuni.example" not in repr(openai_config)


def test_main_accepts_a_prompt_argument(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    monkeypatch.setattr(
        main_module,
        "run",
        lambda prompt: AgentOutcome(output=f"answer to {prompt}", artifacts=()),
    )

    main_module.main(["question"])

    assert capsys.readouterr().out == "answer to question\n"


def test_run_returns_artifacts_published_during_the_one_off_run(
    monkeypatch: Any,
    make_table_artifact: Any,
) -> None:
    config = cast(
        AgentConfig,
        SimpleNamespace(
            catalog_url="https://catalog.example",
            catalog_token="token",
            catalog_name="analytics",
            schema_name="bronze",
            table_column_page_size=20,
            temperature=0.0,
            request_limit=3,
            enabled_capabilities=("query", "table_outputs"),
            create_model=lambda: "model",
        ),
    )
    monkeypatch.setattr(main_module, "get_spark_session", lambda _config: object())

    def run_sync(_prompt: str, **kwargs: Any) -> SimpleNamespace:
        artifact = make_table_artifact([{"value": 1}])
        kwargs["deps"].artifacts.publish(artifact)
        return SimpleNamespace(output="answer")

    monkeypatch.setattr(
        main_module,
        "create_agent",
        lambda capabilities: SimpleNamespace(run_sync=run_sync),
    )

    outcome = main_module.run("question", config)

    assert outcome.output == "answer"
    assert len(outcome.artifacts) == 1
    assert isinstance(outcome.artifacts[0], TableArtifact)
