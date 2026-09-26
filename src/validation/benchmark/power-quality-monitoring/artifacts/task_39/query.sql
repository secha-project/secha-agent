WITH ev AS (
    SELECT
        window(timestamp, '15 minutes') AS time_interval,
        AVG(pw) / 1000 * 0.25 AS ev_kwh
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name = 'KampusareenaEVCharging'
        AND to_date(timestamp) = DATE '2025-03-02'
    GROUP BY window(timestamp, '15 minutes')
),

solar AS (
    SELECT
        window(timestamp, '15 minutes') AS time_interval,
        AVG(pw) / 1000 * 0.25 AS solar_kwh
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name = 'KampusareenaSolarPVPlant'
        AND to_date(timestamp) = DATE '2025-03-02'
    GROUP BY window(timestamp, '15 minutes')
)

SELECT
    ev.time_interval.start AS time_start,
    ev.time_interval.end AS time_end,
    ev.ev_kwh,
    solar.solar_kwh
FROM ev
JOIN solar
  ON ev.time_interval = solar.time_interval
WHERE ev.ev_kwh > 0
  AND solar.solar_kwh > 0
  AND solar.solar_kwh / ev.ev_kwh >= 0.25
ORDER BY time_start
