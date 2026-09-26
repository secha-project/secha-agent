SELECT window(timestamp, '60 seconds').end AS time,
    AVG(u1l1v) AS u1l1v,
    AVG(u2l1) AS u2l1,
    AVG(u3l1) AS u3l1,
    AVG(u4l1) AS u4l1,
    AVG(u5l1) AS u5l1,
    AVG(u6l1) AS u6l1,
    AVG(u7l1) AS u7l1,
    AVG(u8l1) AS u8l1,
    AVG(u9l1) AS u9l1,
    AVG(u10l1) AS u10l1,
    AVG(u11l1) AS u11l1,
    AVG(u12l1) AS u12l1,
    AVG(u13l1) AS u13l1,
    AVG(u14l1) AS u14l1,
    AVG(u15l1) AS u15l1,
    AVG(u16l1) AS u16l1,
    AVG(u17l1) AS u17l1,
    AVG(u18l1) AS u18l1,
    AVG(u19l1) AS u19l1,
    AVG(u20l1) AS u20l1
FROM analytics.bronze.power_quality_measurements_procem
WHERE device_name = 'KampusareenaSolarPVPlant'
    AND timestamp >= '2025-03-01T12:00:00'
    AND timestamp < '2025-03-01T17:00:00'
GROUP BY window(timestamp, '60 seconds')
ORDER BY time
