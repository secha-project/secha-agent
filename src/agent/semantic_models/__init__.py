"""Registry for code-defined Boring Semantic Layer models."""

from collections.abc import Mapping

from boring_semantic_layer import SemanticModel
from ibis import BaseBackend

from .thd import build_thd_metric


def build_semantic_models(
    backend: BaseBackend,
) -> Mapping[str, SemanticModel]:
    """Build the semantic model registry for one application backend."""
    thd_metric = build_thd_metric(backend)
    return {"voltage_thd_10m": thd_metric}
