WITH minute_values AS (
    SELECT
        timestamp,
        dl1 * u1l1 / 100.0 AS l1_v,
        dl2 * u1l2 / 100.0 AS l2_v,
        dl3 * u1l3 / 100.0 AS l3_v
    FROM analytics.bronze.power_quality_measurements_electrix
    WHERE device_id = 'device_156'
      AND rec_source = 'full'
      AND timestamp >= TIMESTAMP '2025-03-24 00:00:00Z'
      AND timestamp < TIMESTAMP '2025-03-25 00:00:00Z'
),
ten_minute_values AS (
    SELECT
        window(timestamp, '10 minutes').end AS ten_minute_window_end,
        AVG(l1_v) AS l1_v,
        AVG(l2_v) AS l2_v,
        AVG(l3_v) AS l3_v
    FROM minute_values
    GROUP BY window(timestamp, '10 minutes').end
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
    WHERE ten_minute_window_end > TIMESTAMP '2025-03-24 00:00:00Z'
      AND ten_minute_window_end <= TIMESTAMP '2025-03-25 00:00:00Z'
)
SELECT
    phase,
    ROUND(AVG(distortion_v), 4) AS mean_voltage_harmonic_distortion_v
FROM phase_values
GROUP BY phase
ORDER BY phase;
