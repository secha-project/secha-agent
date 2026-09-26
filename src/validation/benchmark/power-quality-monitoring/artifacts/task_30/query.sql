WITH five_minute_components AS (
    SELECT
        WINDOW(timestamp, '5 minutes').start AS interval_start,
        AVG(GREATEST(pw / 1000.0, 0.0)) AS mean_positive_power_kw,
        AVG(LEAST(pw / 1000.0, 0.0)) AS mean_negative_power_kw
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name = 'MarjamakiBatteryStorage02'
      AND timestamp >= TIMESTAMP '2025-03-05 12:00:00'
      AND timestamp < TIMESTAMP '2025-03-05 16:00:00'
      AND pw IS NOT NULL
    GROUP BY
        WINDOW(timestamp, '5 minutes')
)
SELECT
    interval_start,
    power_direction,
    mean_active_power_kw
FROM five_minute_components
LATERAL VIEW STACK(
    2,
    'Positive', mean_positive_power_kw,
    'Negative', mean_negative_power_kw
) AS power_direction, mean_active_power_kw
ORDER BY
    interval_start,
    power_direction
