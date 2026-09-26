SELECT device_identifier
FROM analytics.bronze.device_metadata
WHERE LOWER(description) LIKE '%kampusareena%'
  AND LOWER(description) LIKE '%fcr-n reserve unit pool%'
ORDER BY device_identifier
