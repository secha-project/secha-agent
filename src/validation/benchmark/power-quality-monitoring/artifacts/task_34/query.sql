WITH five_minute_power AS (
    SELECT
        window(`timestamp`, '5 minutes').start AS interval_start,
        AVG(CASE
            WHEN device_name = 'KampusareenaTenants' THEN pw
        END) / 1000.0 AS tenant_demand_kw,
        AVG(CASE
            WHEN device_name = 'KampusareenaMaintenance' THEN pw
        END) / 1000.0 AS shared_services_demand_kw,
        GREATEST(
            AVG(CASE
                WHEN device_name = 'KampusareenaSolarPVPlant' THEN pw
            END) / 1000.0,
            0.0
        ) AS solar_generation_kw
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name IN (
        'KampusareenaTenants',
        'KampusareenaMaintenance',
        'KampusareenaSolarPVPlant'
    )
      AND `timestamp` >= TIMESTAMP '2025-03-24 08:00:00'
      AND `timestamp` <  TIMESTAMP '2025-03-24 16:00:00'
      AND pw IS NOT NULL
    GROUP BY window(`timestamp`, '5 minutes')
),

matched_intervals AS (
    SELECT
        tenant_demand_kw + shared_services_demand_kw AS total_demand_kw,
        solar_generation_kw
    FROM five_minute_power
    WHERE tenant_demand_kw IS NOT NULL
      AND shared_services_demand_kw IS NOT NULL
      AND solar_generation_kw IS NOT NULL
)

SELECT
    100.0 * SUM(LEAST(solar_generation_kw, total_demand_kw))
        / SUM(total_demand_kw) AS demand_supplied_percentage
FROM matched_intervals
