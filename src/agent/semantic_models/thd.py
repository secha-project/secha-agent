import ibis
from boring_semantic_layer import Dimension, Measure, SemanticModel, to_semantic_table
from ibis import BaseBackend


def _thd(t, quantity: str, phase: str, fundamental_unit: str):
    harmonic_power = (t[f"{quantity}2{phase}"] ** 2).mean()
    for order in range(3, 21):
        harmonic_power += (t[f"{quantity}{order}{phase}"] ** 2).mean()
    fundamental = (t[f"{quantity}1{phase}{fundamental_unit}"] ** 2).mean().sqrt()
    return 100 * harmonic_power.sqrt() / fundamental


def _thd_measure(
    quantity_name: str,
    column_prefix: str,
    fundamental_unit: str,
    phase: str,
) -> Measure:
    return Measure(
        expr=lambda t: _thd(t, column_prefix, phase, fundamental_unit),
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
                    t.timestamp.cast("timestamp").bucket(minutes=10)
                    + ibis.interval(minutes=10)
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
            voltage_thd_2_20_l1_pct=_thd_measure("Voltage", "u", "v", "l1"),
            voltage_thd_2_20_l2_pct=_thd_measure("Voltage", "u", "v", "l2"),
            voltage_thd_2_20_l3_pct=_thd_measure("Voltage", "u", "v", "l3"),
            current_thd_2_20_l1_pct=_thd_measure("Current", "i", "a", "l1"),
            current_thd_2_20_l2_pct=_thd_measure("Current", "i", "a", "l2"),
            current_thd_2_20_l3_pct=_thd_measure("Current", "i", "a", "l3"),
        )
    )
