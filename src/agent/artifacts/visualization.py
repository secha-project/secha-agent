from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

VEGA_LITE_MEDIA_TYPE = "application/vnd.vegalite.v6+json"


@dataclass(frozen=True)
class VisualizationContent:
    """Portable, self-contained Vega-Lite output."""

    spec_json: bytes
    kind: Literal["visualization"] = field(default="visualization", init=False)
    media_type: Literal["application/vnd.vegalite.v6+json"] = field(
        default=VEGA_LITE_MEDIA_TYPE,
        init=False,
    )

    def to_spec(self) -> dict[str, Any]:
        value = json.loads(self.spec_json)
        if not isinstance(value, dict):
            raise TypeError("Saved Vega-Lite content must be a JSON object.")
        return value
