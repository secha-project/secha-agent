SELECT 2 * (end_reading.`ep+kwh` - start_reading.`ep+kwh`) AS consumed_kwh
FROM analytics.bronze.power_quality_measurements_procem AS start_reading
JOIN analytics.bronze.power_quality_measurements_procem AS end_reading
    ON end_reading.device_name = start_reading.device_name
WHERE start_reading.device_name = 'KampusareenaTenants'
    AND start_reading.`timestamp` >= TIMESTAMP '2025-03-01 00:00:00'
    AND start_reading.`timestamp` < TIMESTAMP '2025-03-01 00:00:01'
    AND end_reading.`timestamp` >= TIMESTAMP '2025-03-02 00:00:00'
    AND end_reading.`timestamp` < TIMESTAMP '2025-03-02 00:00:01'
