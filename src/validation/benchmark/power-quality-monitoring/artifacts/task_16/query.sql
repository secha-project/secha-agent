SELECT MAX_BY(timestamp, pw) timestamp, MAX(pw)/1000.0 as max_pw_kw
FROM analytics.bronze.power_quality_measurements_electrix
WHERE device_id = 'device_156'
    AND timestamp >= '2025-03-01T00:00:00'
    AND timestamp < '2025-03-02T00:00:00'
