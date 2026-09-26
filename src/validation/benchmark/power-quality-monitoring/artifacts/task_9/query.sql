SELECT ROUND(
    100.0 * COUNT_IF(
        pstl1 IS NOT NULL AND pstl1 <> -1
        AND pstl2 IS NOT NULL AND pstl2 <> -1
        AND pstl3 IS NOT NULL AND pstl3 <> -1
    ) / NULLIF(COUNT(*), 0),
    2
) AS records_with_three_phase_flicker_pct
FROM analytics.bronze.power_quality_measurements_electrix
WHERE device_id = 'device_21'
  AND timestamp >= TIMESTAMP '2025-03-05 00:00:00'
  AND timestamp < TIMESTAMP '2025-03-06 00:00:00'
