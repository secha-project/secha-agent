WITH battery_ratings AS (
    SELECT *
    FROM VALUES
        ('MarjamakiBatteryStorage01', 1600.0),
        ('MarjamakiBatteryStorage02', 2400.0)
        AS ratings(device_name, maximum_power_kw)
),

minute_power AS (
    SELECT
        window(measurements.`timestamp`, '1 minute').start AS minute_start,
        measurements.device_name,
        ratings.maximum_power_kw,
        AVG(measurements.pw) / 1000.0 AS mean_power_kw
    FROM analytics.bronze.power_quality_measurements_procem AS measurements
    INNER JOIN battery_ratings AS ratings
        ON ratings.device_name = measurements.device_name
    WHERE measurements.`timestamp` >= TIMESTAMP '2025-03-05 00:00:00'
      AND measurements.`timestamp` <  TIMESTAMP '2025-03-06 00:00:00'
      AND measurements.pw IS NOT NULL
    GROUP BY
        window(measurements.`timestamp`, '1 minute'),
        measurements.device_name,
        ratings.maximum_power_kw
),

matched_minutes AS (
    SELECT
        minute_start,
        MAX(
            CASE
                WHEN device_name = 'MarjamakiBatteryStorage01'
                THEN mean_power_kw
            END
        ) AS bess_1_power_kw,
        MAX(
            CASE
                WHEN device_name = 'MarjamakiBatteryStorage01'
                THEN maximum_power_kw
            END
        ) AS bess_1_maximum_power_kw,
        MAX(
            CASE
                WHEN device_name = 'MarjamakiBatteryStorage02'
                THEN mean_power_kw
            END
        ) AS bess_2_power_kw,
        MAX(
            CASE
                WHEN device_name = 'MarjamakiBatteryStorage02'
                THEN maximum_power_kw
            END
        ) AS bess_2_maximum_power_kw
    FROM minute_power
    GROUP BY minute_start
    HAVING COUNT(DISTINCT device_name) = 2
),

jointly_active_minutes AS (
    SELECT
        minute_start,
        bess_1_power_kw,
        bess_2_power_kw
    FROM matched_minutes
    WHERE ABS(bess_1_power_kw) > 0.01 * bess_1_maximum_power_kw
      AND ABS(bess_2_power_kw) > 0.01 * bess_2_maximum_power_kw
),

hourly_same_direction AS (
    SELECT
        window(minute_start, '1 hour').start AS hour_start,
        100.0 * AVG(
            CASE
                WHEN bess_1_power_kw * bess_2_power_kw > 0 THEN 1.0
                ELSE 0.0
            END
        ) AS same_direction_percentage
    FROM jointly_active_minutes
    GROUP BY window(minute_start, '1 hour')
),

all_hours AS (
    SELECT EXPLODE(
        SEQUENCE(
            TIMESTAMP '2025-03-05 00:00:00',
            TIMESTAMP '2025-03-05 23:00:00',
            INTERVAL 1 HOUR
        )
    ) AS hour_start
)

SELECT
    DATE_FORMAT(all_hours.hour_start, 'HH:mm') AS utc_hour,
    hourly_same_direction.same_direction_percentage
FROM all_hours
LEFT JOIN hourly_same_direction
    ON hourly_same_direction.hour_start = all_hours.hour_start
ORDER BY all_hours.hour_start
