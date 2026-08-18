"""Registry for code-defined Boring Semantic Layer models."""

from collections.abc import Mapping

from boring_semantic_layer import SemanticModel
from ibis import BaseBackend

from .thd import build_harmonic_distortion_model


def build_semantic_models(
    backend: BaseBackend,
) -> Mapping[str, SemanticModel]:
    """Build the semantic model registry for one application backend."""
    harmonic_distortion = build_harmonic_distortion_model(backend)
    return {"harmonic_distortion_10m": harmonic_distortion}
