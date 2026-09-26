WITH readings AS (
    SELECT
        timestamp,
        pw,
        CASE
            WHEN pw < 500 THEN 1
            ELSE 0
        END AS is_idle
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name = 'KampusareenaEVCharging'
      AND TO_DATE(timestamp) = DATE '2025-03-02'
      AND pw IS NOT NULL
),

changes AS (
    SELECT
        timestamp,
        pw,
        is_idle,
        LAG(is_idle, 1, 0) OVER (
            ORDER BY timestamp
        ) AS previous_idle
    FROM readings
),

numbered AS (
    SELECT
        timestamp,
        pw,
        is_idle,
        SUM(
            CASE
                WHEN is_idle = 1
                 AND previous_idle = 0
                THEN 1
                ELSE 0
            END
        ) OVER (
            ORDER BY timestamp
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS group_number
    FROM changes
),

groups AS (
    SELECT
        timestamp,
        pw,
        CASE
            WHEN is_idle = 1 THEN group_number
            ELSE NULL
        END AS idle_group
    FROM numbered
),

idle_periods AS (
    SELECT
        idle_group,
        MIN(timestamp) AS idle_start,
        MAX(timestamp) AS idle_end,
        timestampdiff(
            MINUTE,
            MIN(timestamp),
            MAX(timestamp)
        ) AS duration_minutes,
        AVG(pw) AS average_pw
    FROM groups
    WHERE idle_group IS NOT NULL
    GROUP BY idle_group
)

SELECT
    idle_start,
    idle_end,
    duration_minutes,
    average_pw/1000 as average_pw_kw
FROM idle_periods
WHERE duration_minutes >= 15
ORDER BY idle_start;
