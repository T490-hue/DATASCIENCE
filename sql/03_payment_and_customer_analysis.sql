-- =================================================
-- PAYMENT AND CUSTOMER ANALYSIS QUERIES
-- =================================================

-- Query to see the distribution of different payment types used by customers.
SELECT
    payment_type,
    COUNT(*) AS count
FROM order_payments
GROUP BY payment_type
ORDER BY count DESC;


-- Query to find the average review score for each payment type.
SELECT
    p.payment_type,
    AVG(r.review_score) AS avg_review_score
FROM order_payments p
JOIN order_reviews r ON p.order_id = r.order_id
GROUP BY p.payment_type;


-- Query to calculate the average shipping cost (freight value) by customer state.
SELECT
    c.customer_state,
    AVG(oi.freight_value) AS avg_freight
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY c.customer_state
ORDER BY avg_freight DESC;
