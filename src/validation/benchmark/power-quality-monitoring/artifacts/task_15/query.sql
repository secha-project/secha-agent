WITH hour_starts AS (
    SELECT EXPLODE(
        SEQUENCE(
            TIMESTAMP '2025-03-05 00:00:00',
            TIMESTAMP '2025-03-06 00:00:00' - INTERVAL 1 HOUR,
            INTERVAL 1 HOUR
        )
    ) AS hour_start
),
solar_coverage AS (
    SELECT
        DATE_TRUNC('hour', timestamp) AS hour_start,
        COUNT(DISTINCT DATE_TRUNC('minute', timestamp)) AS solar_minutes
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name = 'KampusareenaSolarPVPlant'
      AND timestamp >= TIMESTAMP '2025-03-05 00:00:00'
      AND timestamp < TIMESTAMP '2025-03-06 00:00:00'
      AND pw IS NOT NULL
    GROUP BY DATE_TRUNC('hour', timestamp)
),
weather_coverage AS (
    SELECT
        DATE_TRUNC('hour', timestamp) AS hour_start,
        COUNT(DISTINCT DATE_TRUNC('minute', timestamp)) AS irradiance_minutes
    FROM analytics.bronze.weather_measurements_procem
    WHERE device_name = 'KampusareenaWeatherStation'
      AND timestamp >= TIMESTAMP '2025-03-05 00:00:00'
      AND timestamp < TIMESTAMP '2025-03-06 00:00:00'
      AND irradiance_global IS NOT NULL
    GROUP BY DATE_TRUNC('hour', timestamp)
),
price_coverage AS (
    SELECT
        timestamp AS hour_start,
        COUNT_IF(complete AND price_FI IS NOT NULL) AS price_records
    FROM analytics.bronze.electricity_prices_procem
    WHERE device_name = 'ElectricityPrice'
      AND timestamp >= TIMESTAMP '2025-03-05 00:00:00'
      AND timestamp < TIMESTAMP '2025-03-06 00:00:00'
    GROUP BY timestamp
)
SELECT
    hour_starts.hour_start,
    COALESCE(solar_coverage.solar_minutes, 0) AS solar_minutes,
    COALESCE(weather_coverage.irradiance_minutes, 0) AS irradiance_minutes,
    COALESCE(price_coverage.price_records, 0) AS price_records
FROM hour_starts
LEFT JOIN solar_coverage USING (hour_start)
LEFT JOIN weather_coverage USING (hour_start)
LEFT JOIN price_coverage USING (hour_start)
ORDER BY hour_starts.hour_start
