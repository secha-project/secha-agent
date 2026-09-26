WITH ten_minute_values AS (
    SELECT
        ten_minute_window_end,
        MEASURE(voltage_thd_2_20_l1_v) AS voltage_l1,
        MEASURE(voltage_thd_2_20_l2_v) AS voltage_l2,
        MEASURE(voltage_thd_2_20_l3_v) AS voltage_l3,
        MEASURE(current_thd_2_20_l1_a) AS current_l1,
        MEASURE(current_thd_2_20_l2_a) AS current_l2,
        MEASURE(current_thd_2_20_l3_a) AS current_l3
    FROM analytics.bronze.harmonic_distortion_10m
    WHERE device_identifier = 'KampusareenaEVCharging'
      AND ten_minute_window_end > TIMESTAMP '2025-03-01 00:00:00+00:00'
      AND ten_minute_window_end <= TIMESTAMP '2025-03-31 00:00:00+00:00'
    GROUP BY ten_minute_window_end
),

daily_wide AS (
    SELECT
        DATE(ten_minute_window_end - INTERVAL 1 MICROSECOND) AS day,
        AVG(voltage_l1) AS voltage_l1,
        AVG(voltage_l2) AS voltage_l2,
        AVG(voltage_l3) AS voltage_l3,
        AVG(current_l1) AS current_l1,
        AVG(current_l2) AS current_l2,
        AVG(current_l3) AS current_l3
    FROM ten_minute_values
    GROUP BY DATE(ten_minute_window_end - INTERVAL 1 MICROSECOND)
),

daily_means AS (
    SELECT day, quantity, phase, daily_mean
    FROM daily_wide
    LATERAL VIEW STACK(
        6,
        'voltage', 'L1', voltage_l1,
        'voltage', 'L2', voltage_l2,
        'voltage', 'L3', voltage_l3,
        'current', 'L1', current_l1,
        'current', 'L2', current_l2,
        'current', 'L3', current_l3
    ) channels AS quantity, phase, daily_mean
),

baseline_comparison AS (
    SELECT
        *,
        AVG(
            CASE
                WHEN day BETWEEN DATE '2025-03-01' AND DATE '2025-03-07'
                THEN daily_mean
            END
        ) OVER (
            PARTITION BY quantity, phase
        ) AS baseline
    FROM daily_means
)

SELECT
    day AS date,
    quantity,
    phase,
    ROUND(daily_mean / baseline, 4) AS factor
FROM baseline_comparison
WHERE day BETWEEN DATE '2025-03-08' AND DATE '2025-03-30'
  AND (
        daily_mean >= 1.5 * baseline
        OR daily_mean <= 0.5 * baseline
      )
ORDER BY date, quantity, phase;
