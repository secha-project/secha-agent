SELECT 2 * (
    MAX_BY(`ep+kwh`, `timestamp`) - MIN_BY(`ep+kwh`, `timestamp`)
) AS consumed_kwh
FROM analytics.bronze.power_quality_measurements_procem
WHERE device_name = 'KampusareenaTenants'
    AND `timestamp` >= TIMESTAMP '2025-03-01 00:00:00'
    AND `timestamp` < TIMESTAMP '2025-03-02 00:00:00'
