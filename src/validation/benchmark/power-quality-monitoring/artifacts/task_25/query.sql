WITH usable_prices AS (
    SELECT
        TO_DATE(timestamp) AS price_date,
        timestamp AS hour_start,
        price_FI
    FROM analytics.bronze.electricity_prices_procem
    WHERE device_name = 'ElectricityPrice'
      AND timestamp >= TIMESTAMP '2025-03-22 00:00:00'
      AND timestamp < TIMESTAMP '2025-03-29 00:00:00'
      AND complete = TRUE
      AND price_FI IS NOT NULL
),
candidate_windows AS (
    SELECT
        price_date,
        hour_start,
        LEAD(hour_start, 2) OVER (
            PARTITION BY price_date
            ORDER BY hour_start
        ) AS third_hour_start,
        COUNT(*) OVER (
            PARTITION BY price_date
            ORDER BY hour_start
            ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING
        ) AS price_hours,
        AVG(price_FI) OVER (
            PARTITION BY price_date
            ORDER BY hour_start
            ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING
        ) AS three_hour_mean_price_eur_per_mwh
    FROM usable_prices
),
ranked_windows AS (
    SELECT
        price_date,
        hour_start,
        three_hour_mean_price_eur_per_mwh,
        ROW_NUMBER() OVER (
            PARTITION BY price_date
            ORDER BY
                three_hour_mean_price_eur_per_mwh,
                hour_start
        ) AS window_rank
    FROM candidate_windows
    WHERE price_hours = 3
      AND third_hour_start = hour_start + INTERVAL 2 HOURS
)
SELECT
    price_date,
    hour_start AS window_start,
    hour_start + INTERVAL 3 HOURS AS window_end_exclusive,
    three_hour_mean_price_eur_per_mwh
FROM ranked_windows
WHERE window_rank = 1
ORDER BY
    price_date
