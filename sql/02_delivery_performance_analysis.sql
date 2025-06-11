-- =================================================
-- DELIVERY PERFORMANCE ANALYSIS QUERIES
-- =================================================

-- Query to calculate the total delivery time in days for each delivered order.
SELECT
    order_id,
    TIMESTAMPDIFF(DAY, order_purchase_timestamp, order_delivered_customer_date) AS delivery_time_days
FROM orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL;


-- Query to categorize each order as 'Late', 'Early', or 'On-time' by comparing actual vs. estimated delivery.
SELECT
    order_id,
    TIMESTAMPDIFF(DAY, order_estimated_delivery_date, order_delivered_customer_date) AS delivery_diff,
    CASE
        WHEN TIMESTAMPDIFF(DAY, order_estimated_delivery_date, order_delivered_customer_date) > 0 THEN 'Late'
        WHEN TIMESTAMPDIFF(DAY, order_estimated_delivery_date, order_delivered_customer_date) < 0 THEN 'Early'
        ELSE 'On-time'
    END AS delivery_status
FROM orders
WHERE order_status = 'delivered'
  AND order_estimated_delivery_date IS NOT NULL
  AND order_delivered_customer_date IS NOT NULL;


-- Query to analyze the relationship between delivery status and customer review scores.
SELECT
    CASE
        WHEN TIMESTAMPDIFF(DAY, order_estimated_delivery_date, order_delivered_customer_date) > 10 THEN 'Very Late'
        WHEN TIMESTAMPDIFF(DAY, order_estimated_delivery_date, order_delivered_customer_date) > 0 THEN 'Late'
        WHEN TIMESTAMPDIFF(DAY, order_estimated_delivery_date, order_delivered_customer_date) = 0 THEN 'On Time'
        ELSE 'Early'
    END AS delivery_status_category,
    r.review_score,
    COUNT(*) AS frequency
FROM orders o
JOIN order_reviews r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
GROUP BY delivery_status_category, r.review_score
ORDER BY delivery_status_category, r.review_score;
