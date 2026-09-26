WITH five_minute_temperatures AS (
    SELECT
        window(timestamp, '5 minutes').start AS interval_start,
        AVG(temperature) AS mean_temperature_c
    FROM analytics.bronze.weather_measurements_procem
    WHERE timestamp >= TIMESTAMP '2025-03-24 00:00:00'
      AND timestamp < TIMESTAMP '2025-03-25 00:00:00'
      AND temperature IS NOT NULL
    GROUP BY
        window(timestamp, '5 minutes')
)
SELECT
    interval_start,
    FLOOR(mean_temperature_c) + 0.5 AS temperature_bin_midpoint_c
FROM five_minute_temperatures
ORDER BY
    interval_start
