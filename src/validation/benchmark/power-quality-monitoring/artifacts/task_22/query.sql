WITH rolling AS (
    SELECT
        device_name,
        timestamp,
        pw,
        MIN(pw) OVER (
            ORDER BY timestamp
            RANGE BETWEEN INTERVAL 5 MINUTES PRECEDING AND CURRENT ROW
        ) AS pw_rolling_min,
        MAX(pw) OVER (
            ORDER BY timestamp
            RANGE BETWEEN INTERVAL 5 MINUTES PRECEDING AND CURRENT ROW
        ) AS pw_rolling_max
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name = 'KampusareenaEVCharging'
        AND to_date(timestamp) = DATE '2025-03-01'
),

detector AS (
    SELECT
        *,
        CASE
            WHEN pw_rolling_min <> 0
                AND ABS(pw_rolling_max - pw_rolling_min)/pw_rolling_min > 0.13
            THEN 1
            ELSE 0
        END AS flagged
    FROM rolling
)

SELECT
    timestamp,
    pw
FROM detector
WHERE flagged = 1
ORDER BY device_name, timestamp
