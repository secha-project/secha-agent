from typing import Literal


type CapabilityName = Literal["catalog", "query"]
DEFAULT_CAPABILITIES: tuple[CapabilityName, ...] = ("catalog", "query")
