WITH procem_orders AS (
    SELECT *
    FROM VALUES
        (2), (3), (4), (5), (6), (7), (8), (9), (10),
        (11), (12), (13), (14), (15), (16), (17), (18), (19), (20)
    AS procem(harmonic_order)
),
electrix_orders AS (
    SELECT *
    FROM VALUES
        (3), (5), (7), (9), (11), (13), (15), (17), (19),
        (21), (23), (25), (27), (29), (31), (33), (35), (37), (39)
    AS electrix(harmonic_order)
),
shared_orders AS (
    SELECT procem_orders.harmonic_order
    FROM procem_orders
    INNER JOIN electrix_orders USING (harmonic_order)
)
SELECT
    (SELECT COUNT(*) FROM procem_orders) AS procem_harmonic_orders,
    (SELECT COUNT(*) FROM electrix_orders) AS electrix_harmonic_orders,
    (SELECT COUNT(*) FROM shared_orders) AS shared_harmonic_orders
