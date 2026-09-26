WITH net_energy AS (
    SELECT
        device_name,
        MEASURE(net_active_energy_kwh) AS net_active_energy_kwh
    FROM analytics.bronze.recorded_net_active_energy
    WHERE device_name LIKE 'Kampusareena%'
      AND measured_at >= TIMESTAMP '2025-03-24 00:00:00+00:00'
      AND measured_at < TIMESTAMP '2025-03-31 00:00:00+00:00'
    GROUP BY device_name
),
flow_conventions AS (
    SELECT
        device_name,
        SIGN(net_active_energy_kwh)
            * CASE WHEN device_name = 'KampusareenaSolarPVPlant' THEN -1 ELSE 1 END
            AS convention
    FROM net_energy
    WHERE net_active_energy_kwh <> 0
),
majority_convention AS (
    SELECT SIGN(SUM(convention)) AS convention
    FROM flow_conventions
)
SELECT meter.device_name
FROM flow_conventions AS meter
CROSS JOIN majority_convention AS majority
WHERE majority.convention <> 0
  AND meter.convention <> majority.convention
ORDER BY meter.device_name;
