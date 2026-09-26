WITH direction_components AS (
    SELECT
        AVG(SIN(RADIANS(wind_direction))) AS mean_sine,
        AVG(COS(RADIANS(wind_direction))) AS mean_cosine
    FROM analytics.bronze.weather_measurements_procem
    WHERE timestamp >= TIMESTAMP '2025-03-24 12:00:00'
      AND timestamp < TIMESTAMP '2025-03-24 13:00:00'
      AND complete = TRUE
      AND wind_direction IS NOT NULL
),
mean_direction AS (
    SELECT
        mean_sine,
        mean_cosine,
        SQRT(
            mean_sine * mean_sine
            + mean_cosine * mean_cosine
        ) AS mean_vector_length
    FROM direction_components
)
SELECT
    CASE
        WHEN mean_vector_length = 0 THEN CAST(NULL AS DOUBLE)
        ELSE ROUND(
            PMOD(
                DEGREES(ATAN2(mean_sine, mean_cosine)) + 360.0,
                360.0
            ),
            6
        )
    END AS average_wind_direction_degrees
FROM mean_direction
