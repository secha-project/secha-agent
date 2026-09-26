WITH solar_five_minute AS (
    SELECT
        window(`timestamp`, '5 minutes').start AS interval_start,
        GREATEST(AVG(pw), 0.0) / 1000.0 AS solar_power_kw
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name = 'KampusareenaSolarPVPlant'
      AND `timestamp` >= TIMESTAMP '2025-03-24 08:00:00'
      AND `timestamp` <  TIMESTAMP '2025-03-24 16:00:00'
      AND pw IS NOT NULL
    GROUP BY window(`timestamp`, '5 minutes')
),

irradiance_five_minute AS (
    SELECT
        window(`timestamp`, '5 minutes').start AS interval_start,
        AVG(irradiance_global) AS irradiance_w_per_m2
    FROM analytics.bronze.weather_measurements_procem
    WHERE `timestamp` >= TIMESTAMP '2025-03-24 08:00:00'
      AND `timestamp` <  TIMESTAMP '2025-03-24 16:00:00'
      AND irradiance_global IS NOT NULL
    GROUP BY window(`timestamp`, '5 minutes')
)

SELECT
    solar.interval_start,
    irradiance.irradiance_w_per_m2,
    solar.solar_power_kw,
    HOUR(solar.interval_start) AS utc_hour
FROM solar_five_minute AS solar
INNER JOIN irradiance_five_minute AS irradiance
    ON irradiance.interval_start = solar.interval_start
ORDER BY solar.interval_start
