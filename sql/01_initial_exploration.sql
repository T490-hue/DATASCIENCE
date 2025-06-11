-- =================================================
-- INITIAL EXPLORATION QUERIES
-- =================================================

-- Query to count the number of records in each primary table to understand the data scale.
SELECT 'orders' AS table_name, COUNT(*) AS row_count FROM orders
UNION ALL
SELECT 'order_items', COUNT(*) FROM order_items
UNION ALL
SELECT 'order_reviews', COUNT(*) FROM order_reviews
UNION ALL
SELECT 'order_payments', COUNT(*) FROM order_payments
UNION ALL
SELECT 'customers', COUNT(*) FROM customers;


-- Query to calculate the distribution and probability of each review score.
SELECT
    review_score,
    COUNT(*) AS frequency,
    COUNT(*) / (SELECT COUNT(*) FROM order_reviews) AS probability
FROM order_reviews
GROUP BY review_score
ORDER BY review_score;
