WITH meter_metrics AS (
    SELECT
        device_id,
        MEASURE(mean_absolute_distortion_difference_l1_v) AS l1_discrepancy_v,
        MEASURE(mean_absolute_distortion_difference_l2_v) AS l2_discrepancy_v,
        MEASURE(mean_absolute_distortion_difference_l3_v) AS l3_discrepancy_v
    FROM analytics.bronze.voltage_distortion_consistency
    WHERE measured_at >= TIMESTAMP '2025-03-01 00:00:00+00:00'
      AND measured_at < TIMESTAMP '2025-03-08 00:00:00+00:00'
    GROUP BY device_id
),
phase_discrepancies AS (
    SELECT
        device_id,
        phase,
        mean_absolute_discrepancy_v
    FROM meter_metrics
    LATERAL VIEW STACK(
        3,
        'L1', l1_discrepancy_v,
        'L2', l2_discrepancy_v,
        'L3', l3_discrepancy_v
    ) phases AS phase, mean_absolute_discrepancy_v
)
SELECT
    device_id,
    phase,
    mean_absolute_discrepancy_v
FROM phase_discrepancies
ORDER BY mean_absolute_discrepancy_v DESC
LIMIT 1;
