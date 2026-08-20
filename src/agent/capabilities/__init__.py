"""Agent capabilities and their tools."""

from collections.abc import Iterable
from typing import Annotated

from pydantic import AfterValidator

from . import catalog, query, semantic_layer, table_results, visualization

# This is the canonical order for capability instructions and tools. Keep the
# semantic layer first so governed measures precede catalog and SQL fallbacks.
CAPABILITIES = (semantic_layer, catalog, query, table_results, visualization)


def validate(enabled: Iterable[str]) -> frozenset[str]:
    """Return one valid capability selection."""
    selected = frozenset(enabled)
    unknown = selected - {capability.NAME for capability in CAPABILITIES}
    if unknown:
        raise ValueError(f"Unknown capabilities: {sorted(unknown)}")

    result_producers = {"semantic_layer", "query"}
    if result_producers & selected and "table_results" not in selected:
        producers = ", ".join(sorted(result_producers & selected))
        raise ValueError(f"The {producers} capability requires table_results.")
    if "visualization" in selected:
        if "table_results" not in selected:
            raise ValueError("The visualization capability requires table_results.")
        if not result_producers & selected:
            raise ValueError(
                "The visualization capability requires semantic_layer or query."
            )
    return selected


EnabledCapabilities = Annotated[frozenset[str], AfterValidator(validate)]
