WITH wind_by_minute AS (
    SELECT
        window(`timestamp`, '1 minute').start AS minute_start,
        AVG(wind_speed) AS mean_wind_speed_m_per_s
    FROM analytics.bronze.weather_measurements_procem
    WHERE `timestamp` >= TIMESTAMP '2025-03-17 08:00:00'
      AND `timestamp` <  TIMESTAMP '2025-03-17 12:00:00'
      AND wind_speed IS NOT NULL
    GROUP BY window(`timestamp`, '1 minute')
),

ventilation_by_minute AS (
    SELECT
        window(`timestamp`, '1 minute').start AS minute_start,
        AVG(pw) / 1000.0 AS mean_ventilation_power_kw
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name = 'KampusareenaVentilation'
      AND `timestamp` >= TIMESTAMP '2025-03-17 08:00:00'
      AND `timestamp` <  TIMESTAMP '2025-03-17 12:00:00'
      AND pw IS NOT NULL
    GROUP BY window(`timestamp`, '1 minute')
),

same_minute_pairs AS (
    SELECT
        wind.mean_wind_speed_m_per_s,
        ventilation.mean_ventilation_power_kw
    FROM wind_by_minute AS wind
    INNER JOIN ventilation_by_minute AS ventilation
        ON ventilation.minute_start = wind.minute_start
),

five_minute_later_pairs AS (
    SELECT
        wind.mean_wind_speed_m_per_s,
        ventilation.mean_ventilation_power_kw
    FROM wind_by_minute AS wind
    INNER JOIN ventilation_by_minute AS ventilation
        ON ventilation.minute_start = wind.minute_start + INTERVAL 5 MINUTES
)

SELECT
    (
        SELECT CORR(
            mean_wind_speed_m_per_s,
            mean_ventilation_power_kw
        )
        FROM same_minute_pairs
    ) AS same_minute_correlation,
    (
        SELECT CORR(
            mean_wind_speed_m_per_s,
            mean_ventilation_power_kw
        )
        FROM five_minute_later_pairs
    ) AS five_minute_later_correlation
