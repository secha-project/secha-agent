SELECT
    CAST(MIN(timestamp) AS DATE) AS start_date,
    CAST(MAX(timestamp) AS DATE) AS end_date
FROM analytics.bronze.power_quality_measurements_procem
