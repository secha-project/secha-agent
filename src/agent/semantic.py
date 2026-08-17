from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import ibis
from boring_semantic_layer import SemanticModel, to_untagged
from ibis import BaseBackend
from pyspark.sql import SparkSession

from .semantic_models import build_semantic_models


@dataclass(frozen=True)
class SemanticRuntime:
    """Semantic models bound to the application's authenticated Spark session."""

    backend: BaseBackend
    models: Mapping[str, SemanticModel]

    def get_model(self, model_name: str) -> SemanticModel:
        try:
            return self.models[model_name]
        except KeyError:
            available = ", ".join(sorted(self.models)) or "none"
            raise ValueError(
                f"Semantic model {model_name!r} was not found. "
                f"Available models: {available}."
            ) from None

    def compile(self, expression: Any) -> str:
        """Compile a BSL/Ibis expression for execution by the shared Spark session."""
        sql = self.backend.compile(to_untagged(expression))
        if not isinstance(sql, str):
            raise TypeError(
                "The Ibis PySpark backend returned a non-string compiled query."
            )
        return sql


def load_semantic_runtime(
    spark: SparkSession,
) -> SemanticRuntime:
    """Build the application's semantic models against its Spark session."""
    backend = ibis.pyspark.connect(spark)
    models = build_semantic_models(backend)
    return SemanticRuntime(backend=backend, models=models)
