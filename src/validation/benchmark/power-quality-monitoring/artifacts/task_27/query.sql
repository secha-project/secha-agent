WITH usable_prices AS (
    SELECT
        timestamp,
        price_FI
    FROM analytics.bronze.electricity_prices_procem
    WHERE device_name = 'ElectricityPrice'
      AND timestamp >= TIMESTAMP '2025-03-03 00:00:00'
      AND timestamp < TIMESTAMP '2025-03-31 00:00:00'
      AND complete = TRUE
      AND price_FI IS NOT NULL
),
weekday_hour_means AS (
    SELECT
        CASE DAYOFWEEK(timestamp)
            WHEN 2 THEN 'Monday'
            WHEN 3 THEN 'Tuesday'
            WHEN 4 THEN 'Wednesday'
            WHEN 5 THEN 'Thursday'
            WHEN 6 THEN 'Friday'
            WHEN 7 THEN 'Saturday'
            WHEN 1 THEN 'Sunday'
        END AS weekday,
        MOD(DAYOFWEEK(timestamp) + 5, 7) AS weekday_order,
        HOUR(timestamp) AS hour_order,
        LPAD(CAST(HOUR(timestamp) AS STRING), 2, '0') AS utc_hour,
        AVG(price_FI) AS mean_price_eur_per_mwh
    FROM usable_prices
    GROUP BY
        DAYOFWEEK(timestamp),
        HOUR(timestamp)
)
SELECT
    weekday,
    utc_hour,
    mean_price_eur_per_mwh
FROM weekday_hour_means
ORDER BY
    weekday_order,
    hour_order
