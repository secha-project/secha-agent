from typing import Literal


type CapabilityName = Literal["catalog", "query", "visualization"]
DEFAULT_CAPABILITIES: tuple[CapabilityName, ...] = (
    "catalog",
    "query",
    "visualization",
)
