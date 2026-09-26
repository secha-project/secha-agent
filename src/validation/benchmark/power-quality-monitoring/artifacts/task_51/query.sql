WITH snapshot_metrics AS (
    SELECT
        device_name,
        measured_at AS timestamp_utc,
        MEASURE(equivalent_heating_current_l1_a) AS l1_current_a,
        MEASURE(equivalent_heating_current_l2_a) AS l2_current_a,
        MEASURE(equivalent_heating_current_l3_a) AS l3_current_a
    FROM analytics.bronze.current_harmonic_heating
    WHERE device_name IN ('KampusareenaTenants', 'KampusareenaMaintenance')
      AND measured_at >= TIMESTAMP '2025-03-10 00:00:00+00:00'
      AND measured_at < TIMESTAMP '2025-03-17 00:00:00+00:00'
    GROUP BY device_name, measured_at
),
ranked_readings AS (
    SELECT
        *,
        DENSE_RANK() OVER (
            PARTITION BY device_name
            ORDER BY GREATEST(l1_current_a, l2_current_a, l3_current_a) DESC
        ) AS peak_rank
    FROM snapshot_metrics
),
peak_readings AS (
    SELECT *
    FROM ranked_readings
    WHERE peak_rank = 1
)
SELECT
    device_name,
    timestamp_utc,
    phase,
    ROUND(equivalent_current_a, 1) AS equivalent_current_a,
    ROUND(equivalent_current_a - 1443.4, 1) AS difference_from_nominal_a
FROM peak_readings
LATERAL VIEW STACK(
    3,
    'L1', l1_current_a,
    'L2', l2_current_a,
    'L3', l3_current_a
) phases AS phase, equivalent_current_a
WHERE equivalent_current_a = GREATEST(l1_current_a, l2_current_a, l3_current_a)
ORDER BY device_name, timestamp_utc, phase;
