from __future__ import annotations

import os
from dataclasses import dataclass

import logfire
import streamlit as st

from src.agent.agent import create_agent
from src.agent.catalog import UnityCatalogClient
from src.agent.config import AgentConfig
from src.agent.semantic_runtime import load_semantic_runtime
from src.agent.service import AgentService
from src.agent.spark_runtime import get_spark_session

from .storage.history import History, load_history


@dataclass(frozen=True)
class AppResources:
    agent: AgentService
    history: History


_loaded_resources: AppResources | None = None


@st.cache_resource
def load_resources() -> AppResources:
    global _loaded_resources

    config = AgentConfig()  # type: ignore[call-arg]
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = config.otel_exporter_otlp_endpoint
    logfire.configure(
        send_to_logfire=False,
        service_name=config.logfire_service_name,
        metrics=False,
    )
    logfire.instrument_pydantic_ai()
    spark = get_spark_session(config)

    catalog = UnityCatalogClient(
        config.catalog_url,
        config.catalog_token,
        config.catalog_name,
    )
    semantic = (
        load_semantic_runtime(spark)
        if "semantic_layer" in config.enabled_capabilities
        else None
    )

    _loaded_resources = AppResources(
        agent=AgentService(
            config=config,
            agent=create_agent(config.enabled_capabilities),
            spark=spark,
            catalog=catalog,
            semantic=semantic,
        ),
        history=load_history(),
    )
    return _loaded_resources


def shutdown_resources() -> None:
    if _loaded_resources is not None:
        _loaded_resources.agent.spark.stop()
