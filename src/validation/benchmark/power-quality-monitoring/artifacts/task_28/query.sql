WITH rolling_frequency AS (
    SELECT
        timestamp,
        std(fhz) OVER (
            ORDER BY timestamp
            RANGE BETWEEN INTERVAL 5 MINUTES PRECEDING
                AND CURRENT ROW
        ) AS fhz_std
    FROM analytics.bronze.power_quality_measurements_electrix
    WHERE device_id = 'device_21'
        AND timestamp >= TIMESTAMP '2025-03-01 12:55:00'
        AND timestamp < TIMESTAMP '2025-03-01 14:00:00'
)
SELECT
    timestamp,
    fhz_std
FROM rolling_frequency
WHERE timestamp >= TIMESTAMP '2025-03-01 13:00:00'
ORDER BY timestamp
