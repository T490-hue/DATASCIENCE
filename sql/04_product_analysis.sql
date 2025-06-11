-- =================================================
-- PRODUCT ANALYSIS QUERIES
-- =================================================

-- Query to find the most popular product categories based on the number of items sold.
SELECT
    p.product_category_name,
    COUNT(oi.order_item_id) AS total_items
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_category_name
ORDER BY total_items DESC;
