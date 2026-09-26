WITH observed_devices AS (
    SELECT DISTINCT device_id
    FROM analytics.bronze.power_quality_measurements_electrix
    WHERE timestamp >= TIMESTAMP '2025-03-06 08:00:00'
      AND timestamp < TIMESTAMP '2025-03-06 10:00:00'
)
SELECT
    observed_devices.device_id,
    device_metadata.description
FROM observed_devices
LEFT JOIN analytics.bronze.device_metadata AS device_metadata
    ON device_metadata.table_name = 'analytics.bronze.power_quality_measurements_electrix'
    AND device_metadata.device_identifier = observed_devices.device_id
ORDER BY observed_devices.device_id
