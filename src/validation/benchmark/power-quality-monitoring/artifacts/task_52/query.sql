WITH snapshot_metrics AS (
    SELECT
        measured_at AS sample_time,
        MEASURE(nonfundamental_active_power_l1_w) AS l1_nonfundamental_w,
        MEASURE(nonfundamental_active_power_l2_w) AS l2_nonfundamental_w,
        MEASURE(nonfundamental_active_power_l3_w) AS l3_nonfundamental_w,
        MEASURE(fundamental_active_power_l1_w) AS l1_fundamental_w,
        MEASURE(fundamental_active_power_l2_w) AS l2_fundamental_w,
        MEASURE(fundamental_active_power_l3_w) AS l3_fundamental_w
    FROM analytics.bronze.nonfundamental_active_power
    WHERE device_name = 'KampusareenaEVCharging'
      AND measured_at >= TIMESTAMP '2025-03-24 00:00:00+00:00'
      AND measured_at < TIMESTAMP '2025-03-25 00:00:00+00:00'
    GROUP BY measured_at
),
readings AS (
    SELECT
        sample_time,
        phase,
        nonfundamental_w,
        fundamental_w,
        CASE WHEN nonfundamental_w * fundamental_w < 0
                  AND ABS(nonfundamental_w) > 1
             THEN 1 ELSE 0 END AS is_opposing
    FROM snapshot_metrics
    LATERAL VIEW STACK(
        3,
        'L1', l1_nonfundamental_w, l1_fundamental_w,
        'L2', l2_nonfundamental_w, l2_fundamental_w,
        'L3', l3_nonfundamental_w, l3_fundamental_w
    ) phases AS phase, nonfundamental_w, fundamental_w
),
changes AS (
    SELECT
        *,
        LAG(is_opposing, 1, 0) OVER (
            PARTITION BY phase ORDER BY sample_time
        ) AS previous_opposing
    FROM readings
),
numbered AS (
    SELECT
        *,
        SUM(CASE WHEN is_opposing = 1 AND previous_opposing = 0
                 THEN 1 ELSE 0 END) OVER (
            PARTITION BY phase ORDER BY sample_time
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS run_id
    FROM changes
),
runs AS (
    SELECT
        phase,
        MIN(sample_time) AS start_utc,
        MAX(sample_time) AS end_utc,
        TIMESTAMPDIFF(MICROSECOND, MIN(sample_time), MAX(sample_time)) / 1000000.0 AS duration_seconds,
        AVG(ABS(nonfundamental_w)) AS mean_opposing_w,
        MAX(ABS(nonfundamental_w)) AS peak_opposing_w
    FROM numbered
    WHERE is_opposing = 1
    GROUP BY phase, run_id
    HAVING COUNT(*) >= 2
)
SELECT
    phase,
    start_utc,
    end_utc,
    duration_seconds,
    mean_opposing_w,
    peak_opposing_w
FROM runs
ORDER BY duration_seconds DESC, mean_opposing_w DESC, start_utc, phase
LIMIT 10;
