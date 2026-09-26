SELECT
    corr(pw, dl1) AS corr_l1,
    corr(pw, dl2) AS corr_l2,
    corr(pw, dl3) AS corr_l3
FROM analytics.bronze.power_quality_measurements_electrix
WHERE device_id = 'device_156'
    AND timestamp >= TIMESTAMP '2025-03-02 12:00'
    AND timestamp < TIMESTAMP '2025-03-02 17:00'
