WITH one_second_changes AS (
    SELECT
        DATE_TRUNC('second', timestamp) AS measured_at,
        phase,
        active_power_w / 1000.0
            - LAG(active_power_w / 1000.0) OVER phase_timeline
            AS power_change_kw,
        rms_voltage_v
            - LAG(rms_voltage_v) OVER phase_timeline
            AS voltage_change_v
    FROM analytics.bronze.power_quality_measurements_procem
    LATERAL VIEW STACK(
        3,
        'L1', pl1w, ul1v,
        'L2', pl2w, ul2v,
        'L3', pl3w, ul3v
    ) AS phase, active_power_w, rms_voltage_v
    WHERE device_name = 'KampusareenaEVCharging'
      AND timestamp >= TIMESTAMP '2025-03-27 07:09:59Z'
      AND timestamp < TIMESTAMP '2025-03-27 07:17:00Z'
    WINDOW phase_timeline AS (
        PARTITION BY phase
        ORDER BY timestamp
    )
),

lag_correlations AS (
    SELECT
        power.phase,
        TIMESTAMPDIFF(
            SECOND,
            power.measured_at,
            voltage.measured_at
        ) AS delay_seconds,
        CORR(
            power.power_change_kw,
            voltage.voltage_change_v
        ) AS change_correlation
    FROM one_second_changes AS power
    INNER JOIN one_second_changes AS voltage
        ON voltage.phase = power.phase
       AND voltage.measured_at
           BETWEEN power.measured_at + INTERVAL 1 SECOND
               AND power.measured_at + INTERVAL 5 MINUTES
    WHERE power.measured_at >= TIMESTAMP '2025-03-27 07:10:00Z'
      AND power.measured_at < TIMESTAMP '2025-03-27 07:12:00Z'
    GROUP BY
        power.phase,
        TIMESTAMPDIFF(
            SECOND,
            power.measured_at,
            voltage.measured_at
        )
),

ranked_delays AS (
    SELECT
        phase,
        delay_seconds,
        change_correlation,
        ROW_NUMBER() OVER (
            PARTITION BY phase
            ORDER BY change_correlation ASC NULLS LAST, delay_seconds
        ) AS delay_rank
    FROM lag_correlations
)

SELECT
    phase,
    delay_seconds,
    change_correlation
FROM ranked_delays
WHERE delay_rank = 1
ORDER BY phase
