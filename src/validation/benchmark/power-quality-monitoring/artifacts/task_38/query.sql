SELECT
    window(pq.timestamp, '1 hour').end AS time_hour,
    avg(pq.pw) AS power,
    p.price_FI AS price
FROM analytics.bronze.power_quality_measurements_procem AS pq
JOIN analytics.bronze.electricity_prices_procem AS p
    ON date_trunc('hour', pq.timestamp) = date_trunc('hour', p.timestamp)
WHERE pq.device_name = 'KampusareenaEVCharging'
    AND to_date(pq.timestamp) = DATE '2025-03-05'
GROUP BY
    window(pq.timestamp, '1 hour'),
    p.price_FI
ORDER BY time_hour
