WITH ten_minute_values AS (
    SELECT
        ten_minute_window_end,
        MEASURE(voltage_thd_3_39_l1_v) AS l1_v,
        MEASURE(voltage_thd_3_39_l2_v) AS l2_v,
        MEASURE(voltage_thd_3_39_l3_v) AS l3_v
    FROM analytics.bronze.harmonic_distortion_electrix_10m
    WHERE device_identifier = 'device_156'
      AND ten_minute_window_end > TIMESTAMP '2025-03-24 00:00:00Z'
      AND ten_minute_window_end <= TIMESTAMP '2025-03-25 00:00:00Z'
    GROUP BY ten_minute_window_end
),
phase_values AS (
    SELECT
        phase,
        distortion_v
    FROM ten_minute_values
    LATERAL VIEW STACK(
        3,
        'L1', l1_v,
        'L2', l2_v,
        'L3', l3_v
    ) phases AS phase, distortion_v
)
SELECT
    phase,
    ROUND(AVG(distortion_v), 4) AS mean_voltage_harmonic_distortion_v
FROM phase_values
GROUP BY phase
ORDER BY phase;
