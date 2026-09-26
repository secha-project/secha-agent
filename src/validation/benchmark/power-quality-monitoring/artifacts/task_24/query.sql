WITH target_device AS (
    SELECT
        device_identifier
    FROM analytics.bronze.device_metadata
    WHERE table_name = 'analytics.bronze.power_quality_measurements_electrix'
      AND LOWER(description) LIKE '%tty lempäälä%'
),
one_minute_power AS (
    SELECT
        TO_DATE(measurements.timestamp) AS measurement_date,
        window(measurements.timestamp, '1 minute').start AS minute_start,
        AVG(measurements.pw) / 1000.0 AS mean_power_kw
    FROM analytics.bronze.power_quality_measurements_electrix AS measurements
    INNER JOIN target_device
        ON measurements.device_id = target_device.device_identifier
    WHERE measurements.timestamp >= TIMESTAMP '2025-03-03 00:00:00'
      AND measurements.timestamp < TIMESTAMP '2025-03-10 00:00:00'
      AND measurements.pw IS NOT NULL
      AND (
          (
              HOUR(measurements.timestamp) >= 6
              AND HOUR(measurements.timestamp) < 10
          )
          OR (
              HOUR(measurements.timestamp) >= 16
              AND HOUR(measurements.timestamp) < 20
          )
      )
    GROUP BY
        TO_DATE(measurements.timestamp),
        window(measurements.timestamp, '1 minute')
),
daily_dayparts AS (
    SELECT
        measurement_date,
        AVG(CASE
            WHEN HOUR(minute_start) >= 6
             AND HOUR(minute_start) < 10
            THEN mean_power_kw
        END) AS morning_mean_power_kw,
        AVG(CASE
            WHEN HOUR(minute_start) >= 16
             AND HOUR(minute_start) < 20
            THEN mean_power_kw
        END) AS evening_mean_power_kw
    FROM one_minute_power
    GROUP BY
        measurement_date
)
SELECT
    measurement_date,
    morning_mean_power_kw,
    evening_mean_power_kw,
    ABS(evening_mean_power_kw - morning_mean_power_kw)
        AS absolute_difference_kw
FROM daily_dayparts
WHERE morning_mean_power_kw IS NOT NULL
   OR evening_mean_power_kw IS NOT NULL
ORDER BY
    measurement_date
