SELECT device_id, timestamp, fhz
FROM analytics.bronze.power_quality_measurements_electrix
WHERE timestamp >= TIMESTAMP '2025-03-01 00:00:00+00:00'
  AND timestamp < TIMESTAMP '2025-03-02 00:00:00+00:00'
ORDER BY fhz ASC
LIMIT 1
