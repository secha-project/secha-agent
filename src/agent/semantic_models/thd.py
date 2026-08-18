import ibis
from boring_semantic_layer import Dimension, Measure, SemanticModel, to_semantic_table
from ibis import BaseBackend


def _thd(t, phase: str):
    harmonic_power = (t[f"u2{phase}"] ** 2).mean()
    for order in range(3, 21):
        harmonic_power += (t[f"u{order}{phase}"] ** 2).mean()
    fundamental = (t[f"u1{phase}v"] ** 2).mean().sqrt()
    return 100 * harmonic_power.sqrt() / fundamental


def build_thd_metric(backend: BaseBackend) -> SemanticModel:
    """Build UTC-aligned 10-minute voltage THD metrics."""
    source = backend.table(
        "power_quality_measurements_procem", database=("analytics", "bronze")
    )

    return (
        to_semantic_table(
            source,
            name="voltage_thd_10m",
            description=(
                "10-min voltage THD with hmin=2, hmax=20"
            ),
        )
        .with_dimensions(
            timestamp=Dimension(
                expr=lambda t: (
                    t.timestamp.cast("timestamp").truncate("hour")
                    + ibis.interval(minutes=10)
                    * (t.timestamp.cast("timestamp").minute() // 10 + 1)
                ),
                description="UTC 10-minute interval conclusion timestamp.",
                is_time_dimension=True,
                smallest_time_grain="minute",
            ),
            device_name=Dimension(
                expr=lambda t: t.device_name,
                description="Measurement device name.",
            ),
        )
        .with_measures(
            **{
                f"thd_2_20_l{phase}_pct": Measure(
                    expr=lambda t, phase=f"l{phase}": _thd(t, phase),
                    description=f"10-min THD (%) for voltage phase L{phase}.",
                )
                for phase in range(1, 4)
            }
        )
    )
