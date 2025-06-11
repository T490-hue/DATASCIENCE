-- This snippet defines the logic for aggregating payment information.
-- For each order, it calculates the total payment value, number of payment
-- methods used, the primary payment type, and the maximum number of installments.
-- This block is used as the 'PaymentInfo' Common Table Expression (CTE) in the main query.

SELECT
    order_id,
    COUNT(payment_sequential) AS payment_count, -- Number of payment steps/methods
    MAX(payment_type) AS payment_type, -- Simple aggregation for type
    MAX(payment_installments) AS payment_installments, -- Max installments used
    SUM(payment_value) AS payment_value -- Total payment value
FROM order_payments
GROUP BY order_id;
