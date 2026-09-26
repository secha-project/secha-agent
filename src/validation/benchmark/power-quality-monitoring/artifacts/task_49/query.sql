WITH phase_measures AS (
    SELECT
        MEASURE(mean_absolute_discrepancy_l1_va) AS l1_va,
        MEASURE(mean_absolute_discrepancy_l2_va) AS l2_va,
        MEASURE(mean_absolute_discrepancy_l3_va) AS l3_va
    FROM analytics.bronze.fundamental_power_consistency
    WHERE device_name = 'KampusareenaEVCharging'
      AND measured_at >= TIMESTAMP '2025-03-24 00:00:00+00:00'
      AND measured_at < TIMESTAMP '2025-03-31 00:00:00+00:00'
)
SELECT
    phase,
    ROUND(mean_absolute_discrepancy_va, 4) AS mean_absolute_discrepancy_va
FROM phase_measures
LATERAL VIEW STACK(
    3,
    'L1', l1_va,
    'L2', l2_va,
    'L3', l3_va
) phases AS phase, mean_absolute_discrepancy_va
ORDER BY phase;
