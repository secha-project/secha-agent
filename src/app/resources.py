from __future__ import annotations

import os
from dataclasses import dataclass

import logfire
import streamlit as st
from pydantic_ai import Agent
from pyspark.sql import SparkSession

from src.agent.agent import create_agent
from src.agent.catalog import UnityCatalogClient
from src.agent.config import AgentConfig
from src.agent.deps import Deps
from src.agent.outputs.artifacts import ArtifactCollector
from src.agent.outputs.tables import TableOutputStore
from src.agent.semantic_runtime import SemanticRuntime, load_semantic_runtime
from src.agent.spark_runtime import get_spark_session

from .storage.history import History, load_history


@dataclass(frozen=True)
class AppResources:
    config: AgentConfig
    agent: Agent[Deps, str]
    spark: SparkSession
    catalog: UnityCatalogClient
    semantic: SemanticRuntime | None
    history: History


@st.cache_resource
def load_resources() -> AppResources:
    config = AgentConfig()  # type: ignore[call-arg]
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = config.otel_exporter_otlp_endpoint
    logfire.configure(
        send_to_logfire=False,
        service_name=config.logfire_service_name,
        metrics=False,
    )
    logfire.instrument_pydantic_ai()
    spark = get_spark_session(config)

    return AppResources(
        config=config,
        agent=create_agent(config.enabled_capabilities),
        spark=spark,
        catalog=UnityCatalogClient(
            config.catalog_url,
            config.catalog_token,
            config.catalog_name,
        ),
        semantic=(
            load_semantic_runtime(spark)
            if "semantic_layer" in config.enabled_capabilities
            else None
        ),
        history=load_history(),
    )


def fresh_deps(
    resources: AppResources,
    table_outputs: TableOutputStore,
) -> Deps:
    """Create per-run dependencies around process-level cached resources."""
    return Deps(
        spark=resources.spark,
        catalog=resources.catalog,
        schema_name=resources.config.schema_name,
        table_column_page_size=resources.config.table_column_page_size,
        semantic=resources.semantic,
        table_outputs=table_outputs,
        artifacts=ArtifactCollector(),
    )
