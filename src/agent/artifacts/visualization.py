from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal


@dataclass(frozen=True)
class VisualizationContent:
    """Portable, self-contained Vega-Lite output."""

    MEDIA_TYPE: ClassVar[Literal["application/vnd.vegalite.v6+json"]] = (
        "application/vnd.vegalite.v6+json"
    )

    spec: dict[str, Any]
    kind: Literal["visualization"] = field(default="visualization", init=False)
    media_type: Literal["application/vnd.vegalite.v6+json"] = field(
        default=MEDIA_TYPE,
        init=False,
    )

    @property
    def spec_json(self) -> bytes:
        return (json.dumps(self.spec, indent=2, ensure_ascii=False) + "\n").encode()

    def to_spec(self) -> dict[str, Any]:
        return self.spec.copy()
