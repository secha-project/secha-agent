WITH interval_power AS (
    SELECT
        window(`timestamp`, '15 minutes').start AS interval_start,
        CASE device_name
            WHEN 'KampusareenaEVCharging' THEN 'EV charging'
            WHEN 'KampusareenaSolarPVPlant' THEN 'Solar generation'
        END AS power_series,
        GREATEST(AVG(pw), 0.0) / 1000.0 AS mean_positive_power_kw
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name IN (
        'KampusareenaEVCharging',
        'KampusareenaSolarPVPlant'
    )
      AND `timestamp` >= TIMESTAMP '2025-03-02 00:00:00'
      AND `timestamp` <  TIMESTAMP '2025-03-03 00:00:00'
      AND pw IS NOT NULL
    GROUP BY
        window(`timestamp`, '15 minutes'),
        device_name
),

all_intervals AS (
    SELECT EXPLODE(
        SEQUENCE(
            TIMESTAMP '2025-03-02 00:00:00',
            TIMESTAMP '2025-03-02 23:45:00',
            INTERVAL 15 MINUTES
        )
    ) AS interval_start
),

power_series AS (
    SELECT *
    FROM VALUES
        ('EV charging', 1),
        ('Solar generation', 2)
        AS series(power_series, series_order)
),

plot_grid AS (
    SELECT
        all_intervals.interval_start,
        power_series.power_series,
        power_series.series_order
    FROM all_intervals
    CROSS JOIN power_series
)

SELECT
    plot_grid.interval_start,
    plot_grid.power_series,
    interval_power.mean_positive_power_kw
FROM plot_grid
LEFT JOIN interval_power
    ON interval_power.interval_start = plot_grid.interval_start
   AND interval_power.power_series = plot_grid.power_series
ORDER BY
    plot_grid.interval_start,
    plot_grid.series_order
