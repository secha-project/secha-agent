WITH rolling AS (
    SELECT
        timestamp,
        AVG(pw) OVER (
            ORDER BY timestamp
            RANGE BETWEEN INTERVAL 15 MINUTES PRECEDING
                AND CURRENT ROW
        ) AS rolling_avg_pw
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name = 'KampusareenaEVCharging'
        AND to_date(timestamp) = DATE '2025-03-01'
)

SELECT timestamp, rolling_avg_pw/1000.0
FROM rolling
ORDER BY rolling_avg_pw DESC
LIMIT 1
