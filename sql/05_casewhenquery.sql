
SELECT
    -- === Identifiers (Keep for lookup, EXCLUDE from direct ML features) ===
    o.order_id,
    r.review_id,
    c.customer_id,
    c.customer_unique_id, -- Tracks the customer across multiple orders

    -- === Target Variable(s) ===
    r.review_score,
    CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END AS is_low_score, -- Binary target

    -- === Order Timing Features (Calculated Durations) ===
    TIMESTAMPDIFF(DAY, o.order_purchase_timestamp, o.order_delivered_customer_date) AS delivery_days,
    TIMESTAMPDIFF(DAY, o.order_estimated_delivery_date, o.order_delivered_customer_date) AS delivery_diff_days, -- Lateness/Earliness
    TIMESTAMPDIFF(HOUR, o.order_purchase_timestamp, o.order_approved_at) AS approval_hours,
    TIMESTAMPDIFF(HOUR, o.order_approved_at, o.order_delivered_carrier_date) AS processing_hours,
    TIMESTAMPDIFF(HOUR, o.order_delivered_carrier_date, o.order_delivered_customer_date) AS carrier_hours,

    -- === Review Timing Features ===
    TIMESTAMPDIFF(HOUR, r.review_creation_date, r.review_answer_timestamp) AS review_response_hours -- Review response time

-- (Content from 05_query_joins_and_filters.sql)
FROM order_reviews r
JOIN orders o ON r.order_id = o.order_id
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN PaymentInfo pay ON o.order_id = pay.order_id
LEFT JOIN ItemProductInfo ipi ON o.order_id = ipi.order_id

WHERE o.order_status = 'delivered'
  AND o.order_purchase_timestamp IS NOT NULL
  AND o.order_approved_at IS NOT NULL
  AND o.order_delivered_carrier_date IS NOT NULL
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
  AND r.review_creation_date IS NOT NULL
  AND r.review_answer_timestamp IS NOT NULL;
