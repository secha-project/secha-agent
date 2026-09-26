WITH minute_power AS (
    SELECT
        window(`timestamp`, '1 minute').start AS minute_start,
        GREATEST(AVG(pw), 0.0) / 1000.0 AS mean_positive_power_kw
    FROM analytics.bronze.power_quality_measurements_electrix
    WHERE device_id = 'device_21'
      AND `timestamp` >= TIMESTAMP '2025-03-05 00:00:00'
      AND `timestamp` <  TIMESTAMP '2025-03-06 00:00:00'
      AND pw IS NOT NULL
    GROUP BY window(`timestamp`, '1 minute')
),

hourly_power AS (
    SELECT
        DATE_TRUNC('hour', minute_start) AS hour_start,
        AVG(mean_positive_power_kw) AS mean_positive_charging_power_kw
    FROM minute_power
    GROUP BY DATE_TRUNC('hour', minute_start)
),

price_hours AS (
    SELECT
        `timestamp` AS hour_start,
        MAX(price_FI) AS spot_price_eur_per_mwh
    FROM analytics.bronze.electricity_prices_procem
    WHERE `timestamp` >= TIMESTAMP '2025-03-05 00:00:00'
      AND `timestamp` <  TIMESTAMP '2025-03-06 00:00:00'
      AND complete = TRUE
      AND price_FI IS NOT NULL
    GROUP BY `timestamp`
    HAVING COUNT(DISTINCT price_FI) = 1
)

SELECT
    power.hour_start,
    power.mean_positive_charging_power_kw,
    price.spot_price_eur_per_mwh
FROM hourly_power AS power
INNER JOIN price_hours AS price
    ON price.hour_start = power.hour_start
ORDER BY power.hour_start
