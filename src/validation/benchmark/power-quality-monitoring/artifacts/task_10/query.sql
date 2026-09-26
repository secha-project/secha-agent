SELECT
    COUNT(DISTINCT CASE
        WHEN table_name = 'analytics.bronze.power_quality_measurements_procem'
        THEN device_identifier
    END) AS procem_device_count,
    COUNT(DISTINCT CASE
        WHEN table_name = 'analytics.bronze.power_quality_measurements_electrix'
        THEN device_identifier
    END) AS electrix_device_count
FROM analytics.bronze.device_metadata
