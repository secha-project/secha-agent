"""Agent capabilities and their tools."""

from collections.abc import Iterable
from typing import Annotated

from pydantic import AfterValidator

from . import catalog, query, semantic_layer, table_outputs, visualization

# This is the canonical order for capability instructions and tools. Keep the
# semantic layer first so governed measures precede catalog and SQL fallbacks.
CAPABILITIES = (semantic_layer, catalog, query, table_outputs, visualization)


def validate(enabled: Iterable[str]) -> frozenset[str]:
    """Return one valid capability selection."""
    selected = frozenset(enabled)
    unknown = selected - {capability.NAME for capability in CAPABILITIES}
    if unknown:
        raise ValueError(f"Unknown capabilities: {sorted(unknown)}")

    output_producers = {"semantic_layer", "query"}
    if output_producers & selected and "table_outputs" not in selected:
        producers = ", ".join(sorted(output_producers & selected))
        raise ValueError(f"The {producers} capability requires table_outputs.")
    if "visualization" in selected:
        if "table_outputs" not in selected:
            raise ValueError("The visualization capability requires table_outputs.")
        if not output_producers & selected:
            raise ValueError(
                "The visualization capability requires semantic_layer or query."
            )
    return selected


EnabledCapabilities = Annotated[frozenset[str], AfterValidator(validate)]
