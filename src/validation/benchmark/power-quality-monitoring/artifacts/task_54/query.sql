WITH one_second_frequency AS (
    SELECT
        window.start AS reading_time,
        avg(freq) FILTER (
            WHERE device_name = 'KampusareenaEVCharging'
        ) AS kampusareena_frequency_hz,
        avg(freq) FILTER (
            WHERE device_name = 'MarjamakiSolarPowerPlant01'
        ) AS marjamaki_frequency_hz
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name IN ('KampusareenaEVCharging', 'MarjamakiSolarPowerPlant01')
      AND timestamp >= TIMESTAMP '2025-03-24 00:00:00Z' - INTERVAL 10 SECONDS
      AND timestamp < TIMESTAMP '2025-03-27 00:00:00Z' + INTERVAL 10 SECONDS
    GROUP BY window(timestamp, '1 second')
), candidate_lags AS (
    -- Test every whole-second shift from -10 to +10 seconds.
    SELECT
        lag_seconds,
        lag_seconds * INTERVAL 1 SECOND AS time_shift
    FROM range(-10, 11) AS shifts(lag_seconds)
), correlation_by_lag AS (
    SELECT
        candidate_lags.lag_seconds,
        corr(
            kampusareena.kampusareena_frequency_hz,
            marjamaki.marjamaki_frequency_hz
        ) AS correlation
    FROM one_second_frequency AS kampusareena
    CROSS JOIN candidate_lags
    JOIN one_second_frequency AS marjamaki
        ON marjamaki.reading_time = kampusareena.reading_time + candidate_lags.time_shift
    WHERE kampusareena.reading_time >= TIMESTAMP '2025-03-24 00:00:00Z'
      AND kampusareena.reading_time < TIMESTAMP '2025-03-27 00:00:00Z'
    GROUP BY candidate_lags.lag_seconds
), ranked_lags AS (
    SELECT
        lag_seconds,
        correlation,
        dense_rank() OVER (ORDER BY correlation DESC NULLS LAST) AS correlation_rank
    FROM correlation_by_lag
)
SELECT
    lag_seconds,
    round(correlation, 6) AS correlation
FROM ranked_lags
WHERE correlation_rank = 1
ORDER BY lag_seconds
