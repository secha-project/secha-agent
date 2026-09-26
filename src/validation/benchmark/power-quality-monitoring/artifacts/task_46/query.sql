WITH three_second_thd AS (
    SELECT
        three_second_window_end,
        device_identifier,
        MEASURE(voltage_thd_2_20_l1_v) AS l1_thd_v,
        MEASURE(voltage_thd_2_20_l2_v) AS l2_thd_v,
        MEASURE(voltage_thd_2_20_l3_v) AS l3_thd_v
    FROM analytics.bronze.harmonic_distortion_3s
    WHERE device_identifier IN (
        'KampusareenaEVCharging',
        'KampusareenaMaintenance'
    )
      AND three_second_window_end > TIMESTAMP '2025-03-29 01:08:00'
      AND three_second_window_end <= TIMESTAMP '2025-03-29 01:10:00'
    GROUP BY
        three_second_window_end,
        device_identifier
),

aligned AS (
    SELECT
        reference.l1_thd_v AS reference_l1,
        reference.l2_thd_v AS reference_l2,
        reference.l3_thd_v AS reference_l3,
        peer.l1_thd_v AS peer_l1,
        peer.l2_thd_v AS peer_l2,
        peer.l3_thd_v AS peer_l3
    FROM three_second_thd AS reference
    INNER JOIN three_second_thd AS peer
        USING (three_second_window_end)
    WHERE reference.device_identifier = 'KampusareenaEVCharging'
      AND peer.device_identifier = 'KampusareenaMaintenance'
)

SELECT
    CORR(reference_l1, peer_l1) AS l1_correlation,
    CORR(reference_l2, peer_l2) AS l2_correlation,
    CORR(reference_l3, peer_l3) AS l3_correlation
FROM aligned;
