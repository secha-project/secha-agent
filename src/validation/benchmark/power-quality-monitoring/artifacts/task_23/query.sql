WITH march_measurements AS (
    SELECT
        device_id,
        timestamp,
        CASE
            WHEN pstl1 > 0.25
                OR pstl2 > 0.25
                OR pstl3 > 0.25
            THEN 1
            ELSE 0
        END AS exceeds_threshold
    FROM analytics.bronze.power_quality_measurements_electrix
    WHERE timestamp >= TIMESTAMP '2025-03-01 00:00:00'
      AND timestamp <  TIMESTAMP '2025-04-01 00:00:00'
),

tagged AS (
    SELECT
        *,
        SUM(
            CASE
                WHEN exceeds_threshold = 1 THEN 0
                ELSE 1
            END
        ) OVER (
            PARTITION BY device_id
            ORDER BY timestamp
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS streak_id
    FROM march_measurements
),

streaks AS (
    SELECT
        device_id,
        streak_id,
        COUNT(*) AS consecutive_measurements,
        MIN(timestamp) AS streak_start,
        MAX(timestamp) AS streak_end
    FROM tagged
    WHERE exceeds_threshold = 1
    GROUP BY device_id, streak_id
    HAVING COUNT(*) >= 2
)

SELECT DISTINCT
    device_id,
    streak_start,
    streak_end
FROM streaks
ORDER BY device_id, streak_start
