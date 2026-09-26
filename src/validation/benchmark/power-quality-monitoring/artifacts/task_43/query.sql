WITH hourly_mean_power AS (
    SELECT
        window(`timestamp`, '1 hour').start AS hour_start,
        CASE device_name
            WHEN 'KampusareenaElevator01' THEN 'Elevator 1'
            WHEN 'KampusareenaElevator02' THEN 'Elevator 2'
            WHEN 'KampusareenaWaterCooling01' THEN 'Water cooling 1'
            WHEN 'KampusareenaWaterCooling02' THEN 'Water cooling 2'
        END AS service,
        AVG(pw) / 1000.0 AS mean_active_power_kw
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name IN (
        'KampusareenaElevator01',
        'KampusareenaElevator02',
        'KampusareenaWaterCooling01',
        'KampusareenaWaterCooling02'
    )
      AND `timestamp` >= TIMESTAMP '2025-03-17 08:00:00'
      AND `timestamp` <  TIMESTAMP '2025-03-17 16:00:00'
      AND pw IS NOT NULL
    GROUP BY
        window(`timestamp`, '1 hour'),
        device_name
)

SELECT
    hour_start,
    service,
    mean_active_power_kw
FROM hourly_mean_power
ORDER BY
    hour_start,
    CASE service
        WHEN 'Elevator 1' THEN 1
        WHEN 'Elevator 2' THEN 2
        WHEN 'Water cooling 1' THEN 3
        WHEN 'Water cooling 2' THEN 4
    END
