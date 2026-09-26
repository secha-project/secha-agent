WITH minute_power AS (
    SELECT
        window(`timestamp`, '1 minute').start AS minute_start,
        GREATEST(AVG(pw), 0.0) / 1000.0 AS mean_positive_power_kw
    FROM analytics.bronze.power_quality_measurements_electrix
    WHERE device_id = 'device_156'
      AND `timestamp` >= TIMESTAMP '2025-03-29 22:00:00'
      AND `timestamp` <  TIMESTAMP '2025-03-30 21:00:00'
      AND pw IS NOT NULL
    GROUP BY window(`timestamp`, '1 minute')
),

hourly_energy AS (
    SELECT
        DATE_TRUNC('hour', minute_start) AS hour_start,
        SUM(mean_positive_power_kw) / 60.0 AS positive_energy_kwh
    FROM minute_power
    GROUP BY DATE_TRUNC('hour', minute_start)
),

price_hours AS (
    SELECT
        `timestamp` AS hour_start,
        MAX(price_FI) AS price_eur_per_mwh
    FROM analytics.bronze.electricity_prices_procem
    WHERE `timestamp` >= TIMESTAMP '2025-03-29 22:00:00'
      AND `timestamp` <  TIMESTAMP '2025-03-30 21:00:00'
      AND complete = TRUE
      AND price_FI IS NOT NULL
    GROUP BY `timestamp`
    HAVING COUNT(DISTINCT price_FI) = 1
)

SELECT
    SUM(
        hourly_energy.positive_energy_kwh
        * price_hours.price_eur_per_mwh
        / 1000.0
    ) AS estimated_cost_eur
FROM hourly_energy
INNER JOIN price_hours
    ON price_hours.hour_start = hourly_energy.hour_start
