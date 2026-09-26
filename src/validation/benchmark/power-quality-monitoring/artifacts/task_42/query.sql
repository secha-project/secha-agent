WITH fifteen_minute_demand AS (
    SELECT
        window(`timestamp`, '15 minutes').start AS interval_start,
        CASE device_name
            WHEN 'KampusareenaTenants' THEN 'Tenant spaces'
            WHEN 'KampusareenaMaintenance' THEN 'Shared building services'
        END AS demand_group,
        AVG(pw) / 1000.0 AS mean_active_power_kw
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name IN (
        'KampusareenaTenants',
        'KampusareenaMaintenance'
    )
      AND `timestamp` >= TIMESTAMP '2025-03-17 00:00:00'
      AND `timestamp` <  TIMESTAMP '2025-03-18 00:00:00'
      AND pw IS NOT NULL
    GROUP BY
        window(`timestamp`, '15 minutes'),
        device_name
)

SELECT
    interval_start,
    demand_group,
    mean_active_power_kw
FROM fifteen_minute_demand
ORDER BY
    interval_start,
    CASE demand_group
        WHEN 'Shared building services' THEN 1
        WHEN 'Tenant spaces' THEN 2
    END
