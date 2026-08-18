"""Agent capabilities and their tools."""

from collections.abc import Iterable
from typing import Annotated

from pydantic import AfterValidator

from . import catalog, query, semantic, visualization

CAPABILITIES = (catalog, query, semantic, visualization)


def validate(enabled: Iterable[str]) -> frozenset[str]:
    """Return one valid capability selection."""
    selected = frozenset(enabled)
    unknown = selected - {capability.NAME for capability in CAPABILITIES}
    if unknown:
        raise ValueError(f"Unknown capabilities: {sorted(unknown)}")

    if "semantic" in selected and "query" not in selected:
        raise ValueError("The semantic capability requires query.")
    if "visualization" in selected and not {"catalog", "query"} <= selected:
        raise ValueError("The visualization capability requires catalog and query.")
    return selected


EnabledCapabilities = Annotated[frozenset[str], AfterValidator(validate)]
