WITH one_minute_power AS (
    SELECT
        window(timestamp, '1 minute').start AS minute_start,
        AVG(pw) / 1000.0 AS mean_power_kw
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name = 'MarjamakiBatteryStorage01'
      AND timestamp >= TIMESTAMP '2025-03-05 12:00:00'
      AND timestamp < TIMESTAMP '2025-03-05 16:00:00'
      AND pw IS NOT NULL
    GROUP BY
        window(timestamp, '1 minute')
)
SELECT
    ROUND(
        PERCENTILE_CONT(0.50) WITHIN GROUP (
            ORDER BY ABS(mean_power_kw)
        ),
        6
    ) AS p50_absolute_power_kw,
    ROUND(
        PERCENTILE_CONT(0.90) WITHIN GROUP (
            ORDER BY ABS(mean_power_kw)
        ),
        6
    ) AS p90_absolute_power_kw,
    ROUND(
        PERCENTILE_CONT(0.99) WITHIN GROUP (
            ORDER BY ABS(mean_power_kw)
        ),
        6
    ) AS p99_absolute_power_kw
FROM one_minute_power
