WITH hour_starts AS (
    SELECT EXPLODE(
        SEQUENCE(
            TIMESTAMP '2025-03-01 00:00:00',
            TIMESTAMP '2025-03-09 00:00:00' - INTERVAL 1 HOUR,
            INTERVAL 1 HOUR
        )
    ) AS hour_start
),
prices_by_hour AS (
    SELECT
        timestamp AS hour_start,
        COUNT(*) AS price_rows,
        COUNT_IF(complete AND price_FI IS NOT NULL) AS usable_price_rows
    FROM analytics.bronze.electricity_prices_procem
    WHERE device_name = 'ElectricityPrice'
      AND timestamp >= TIMESTAMP '2025-03-01 00:00:00'
      AND timestamp < TIMESTAMP '2025-03-09 00:00:00'
    GROUP BY timestamp
),
daily_coverage AS (
    SELECT
        DATE(hour_starts.hour_start) AS day,
        SUM(
            CASE
                WHEN price_rows = 1 AND usable_price_rows = 1 THEN 1
                ELSE 0
            END
        ) AS usable_hours
    FROM hour_starts
    LEFT JOIN prices_by_hour USING (hour_start)
    GROUP BY DATE(hour_starts.hour_start)
)
SELECT COUNT(*) AS complete_utc_days
FROM daily_coverage
WHERE usable_hours = 24
