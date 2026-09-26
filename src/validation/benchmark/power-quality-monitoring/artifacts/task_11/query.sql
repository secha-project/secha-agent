SELECT *
FROM analytics.bronze.power_quality_measurements_electrix
WHERE
    device_id = 'device_156'
    AND timestamp >= TIMESTAMP '2025-03-01T12:00:00Z'
    AND timestamp < TIMESTAMP '2025-03-01T18:00:00Z'
