import ibis
from boring_semantic_layer import Dimension, Measure, SemanticModel, to_semantic_table
from ibis import BaseBackend


def _thd(t, quantity: str, phase: str):
    # PROCEM harmonic components are percentages of the fundamental
    harmonic_power = (t[f"{quantity}2{phase}"] ** 2).mean()
    for order in range(3, 21):
        harmonic_power += (t[f"{quantity}{order}{phase}"] ** 2).mean()
    return harmonic_power.sqrt()


def _thd_measure(
    quantity_name: str,
    column_prefix: str,
    phase: str,
) -> Measure:
    return Measure(
        expr=lambda t: _thd(t, column_prefix, phase),
        description=f"{quantity_name} THD for phase {phase.upper()}.",
    )


def build_harmonic_distortion_model(backend: BaseBackend) -> SemanticModel:
    """Build UTC-aligned 10-minute voltage and current THD metrics."""
    source = backend.table(
        "power_quality_measurements_procem", database=("analytics", "bronze")
    )

    return (
        to_semantic_table(
            source,
            name="harmonic_distortion_10m",
            description=(
                "UTC-aligned 10-minute voltage and current THD, calculated "
                "from harmonic orders 2–20 as a percentage of the fundamental."
            ),
        )
        .with_dimensions(
            ten_minute_end=Dimension(
                expr=lambda t: (
                    t.timestamp.cast("timestamp").truncate("hour")
                    + ibis.interval(minutes=10)
                    * (t.timestamp.cast("timestamp").minute() // 10 + 1)
                ),
                description="UTC 10-minute interval conclusion timestamp.",
                is_time_dimension=True,
            ),
            device_name=Dimension(
                expr=lambda t: t.device_name,
                description="Measurement device name.",
            ),
        )
        .with_measures(
            voltage_thd_2_20_l1_pct=_thd_measure("Voltage", "u", "l1"),
            voltage_thd_2_20_l2_pct=_thd_measure("Voltage", "u", "l2"),
            voltage_thd_2_20_l3_pct=_thd_measure("Voltage", "u", "l3"),
            current_thd_2_20_l1_pct=_thd_measure("Current", "i", "l1"),
            current_thd_2_20_l2_pct=_thd_measure("Current", "i", "l2"),
            current_thd_2_20_l3_pct=_thd_measure("Current", "i", "l3"),
        )
    )
