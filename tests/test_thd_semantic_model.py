from datetime import UTC, datetime
from math import sqrt
from typing import Any, cast

import ibis
import pytest
from boring_semantic_layer import to_untagged

from src.agent.semantic_models.thd import build_harmonic_distortion_model


def test_thd_aggregates_harmonics_before_calculating_ratio() -> None:
    data: dict[str, Any] = {
        "timestamp": [
            datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC),
            datetime(2026, 1, 1, 12, 10, 0, tzinfo=UTC),
            datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        ],
        "device_name": ["meter-a", "meter-a", "meter-a", "meter-b"],
    }
    for phase in range(1, 4):
        data[f"u1l{phase}v"] = [1.0, 3.0, 100.0, 100.0]
        data[f"i1l{phase}a"] = [2.0, 6.0, 100.0, 100.0]
        for order in range(2, 21):
            data[f"u{order}l{phase}"] = (
                [1.0, 1.0, 100.0, 100.0] if order == 2 else [0.0, 0.0, 0.0, 0.0]
            )
            data[f"i{order}l{phase}"] = (
                [1.0, 1.0, 100.0, 100.0] if order == 2 else [0.0, 0.0, 0.0, 0.0]
            )

    source = ibis.memtable(data)

    class Backend:
        def table(self, name: str, *, database: object = None):
            assert name == "power_quality_measurements_procem"
            assert database == ("analytics", "bronze")
            return source

    model = build_harmonic_distortion_model(cast(Any, Backend()))
    expression = model.query(
        dimensions=["ten_minute_end", "device_name"],
        measures=["voltage_thd_2_20_l1_pct", "current_thd_2_20_l1_pct"],
        filters=[{"field": "device_name", "operator": "=", "value": "meter-a"}],
        time_range={
            "start": "2026-01-01T12:10:00",
            "end": "2026-01-01T12:10:00",
        },
    )

    result = to_untagged(expression).execute()

    assert str(result["ten_minute_end"].iloc[0]) == "2026-01-01 12:10:00"
    assert result["device_name"].tolist() == ["meter-a"]
    assert result["voltage_thd_2_20_l1_pct"].tolist() == pytest.approx(
        [100 / sqrt(5)]
    )
    assert result["current_thd_2_20_l1_pct"].tolist() == pytest.approx(
        [50 / sqrt(5)]
    )
