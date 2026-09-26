SELECT *
FROM (
    SELECT device_name, timestamp, freq, ul1v, ul2v, ul3v, u12v, u23v, u31v, tdul1, tdul2, tdul3
    FROM analytics.bronze.power_quality_measurements_procem
    WHERE device_name == 'KampusareenaEVCharging'

    UNION ALL

    SELECT device_id, timestamp, fhz, ul1v, ul2v, ul3v, ul12v, ul23v, ul31v, dl1, dl2, dl3
    FROM analytics.bronze.power_quality_measurements_electrix
)
WHERE timestamp >= TIMESTAMP '2025-03-01T12:00:00'
AND timestamp < TIMESTAMP '2025-03-01T12:05:00'
ORDER BY timestamp, device_name
