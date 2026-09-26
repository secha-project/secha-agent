WITH rolling_window AS (
    SELECT
        timestamp,
        pw,
        AVG(pw) OVER (
            ORDER BY timestamp
            RANGE BETWEEN INTERVAL 600 SECONDS PRECEDING
                AND INTERVAL 0.000001 SECONDS PRECEDING
        ) AS rolling_avg
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name = 'KampusareenaEVCharging'
        AND timestamp >= TIMESTAMP '2025-02-28 23:50:00'
        AND timestamp <  TIMESTAMP '2025-03-02 00:00:00'
        AND pw IS NOT NULL
),

flagged AS (
    SELECT
        timestamp,
        pw,
        rolling_avg,
        pw - rolling_avg AS power_change,
        CASE
            WHEN pw > rolling_avg * 1.05
            AND (pw - rolling_avg) >= 2
            THEN 1
            ELSE 0
        END AS exceeds_threshold
    FROM rolling_window
),

threshold_crossings AS (
    SELECT
        *,
        LAG(exceeds_threshold, 1, 0) OVER (
            ORDER BY timestamp
        ) AS previous_exceeds_threshold
    FROM flagged
)

SELECT
    timestamp AS charging_start,
    pw/1000,
    rolling_avg/1000 AS rolling_avg,
    power_change/1000
FROM threshold_crossings
WHERE to_date(timestamp) = DATE '2025-03-01'
  AND exceeds_threshold = 1
  AND previous_exceeds_threshold = 0
ORDER BY charging_start ASC;
