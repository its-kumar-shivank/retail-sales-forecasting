-- =============================================================================
-- Business-insight queries on the BharatMart warehouse.
-- Techniques used: CTEs, window functions (LAG, SUM OVER, DENSE_RANK, NTILE, AVG
-- OVER ROWS frame), ROLLUP, FILTER, conditional aggregation, PERCENTILE_CONT,
-- self-comparison across years.  Run all:  psql -d retail_dw -f sql/02_analytics_queries.sql
-- =============================================================================


-- Q1. Monthly revenue with Month-over-Month and Year-over-Year growth ----------
WITH monthly AS (
    SELECT d.year, d.month, SUM(f.revenue) AS revenue
    FROM fact_sales f JOIN dim_date d USING (date_key)
    GROUP BY d.year, d.month
)
SELECT year, month,
       ROUND(revenue / 1e7, 2)                                                 AS revenue_cr,
       ROUND(100 * (revenue / LAG(revenue, 1)  OVER (ORDER BY year, month) - 1), 1) AS mom_pct,
       ROUND(100 * (revenue / LAG(revenue, 12) OVER (ORDER BY year, month) - 1), 1) AS yoy_pct
FROM monthly
ORDER BY year, month;


-- Q2. Year-to-date running revenue (resets every calendar year) ----------------
SELECT d.year, d.month,
       ROUND(SUM(f.revenue) / 1e7, 2) AS month_cr,
       ROUND(SUM(SUM(f.revenue)) OVER (PARTITION BY d.year ORDER BY d.month) / 1e7, 2) AS ytd_cr
FROM fact_sales f JOIN dim_date d USING (date_key)
GROUP BY d.year, d.month
ORDER BY d.year, d.month;


-- Q3. Top 3 products by revenue in each region (2025) --------------------------
WITH ranked AS (
    SELECT s.region, p.product_name, p.category,
           SUM(f.revenue) AS revenue,
           DENSE_RANK() OVER (PARTITION BY s.region ORDER BY SUM(f.revenue) DESC) AS rnk
    FROM fact_sales f
    JOIN dim_store s   USING (store_key)
    JOIN dim_product p USING (product_key)
    WHERE f.date_key BETWEEN 20250101 AND 20251231           -- partition pruning: scans 2025 only
    GROUP BY s.region, p.product_name, p.category
)
SELECT region, rnk, product_name, category, ROUND(revenue / 1e5, 1) AS revenue_lakh
FROM ranked WHERE rnk <= 3
ORDER BY region, rnk;


-- Q4. ABC (Pareto) analysis: which products make 80% of revenue? ---------------
WITH prod AS (
    SELECT p.product_name, p.category, SUM(f.revenue) AS revenue
    FROM fact_sales f JOIN dim_product p USING (product_key)
    GROUP BY p.product_name, p.category
), cum AS (
    SELECT *,
           SUM(revenue) OVER (ORDER BY revenue DESC) / SUM(revenue) OVER () AS cum_share
    FROM prod
)
SELECT product_name, category,
       ROUND(revenue / 1e7, 2)    AS revenue_cr,
       ROUND(100 * cum_share, 1)  AS cum_share_pct,
       CASE WHEN cum_share <= 0.80 THEN 'A'
            WHEN cum_share <= 0.95 THEN 'B'
            ELSE 'C' END          AS abc_class
FROM cum
ORDER BY revenue DESC;


-- Q5. Store scorecard: revenue, margin, space productivity, quartile ----------
WITH s25 AS (
    SELECT s.store_name, s.store_type, s.region, s.store_size_sqft,
           SUM(f.revenue) AS revenue, SUM(f.profit) AS profit
    FROM fact_sales f JOIN dim_store s USING (store_key)
    WHERE f.date_key BETWEEN 20250101 AND 20251231
    GROUP BY s.store_name, s.store_type, s.region, s.store_size_sqft
)
SELECT store_name, store_type, region,
       ROUND(revenue / 1e7, 2)                         AS revenue_cr,
       ROUND(100 * profit / revenue, 1)                AS margin_pct,
       ROUND(revenue / store_size_sqft)                AS revenue_per_sqft,
       RANK()  OVER (ORDER BY revenue DESC)            AS revenue_rank,
       RANK()  OVER (ORDER BY revenue / store_size_sqft DESC) AS productivity_rank,
       NTILE(4) OVER (ORDER BY profit DESC)            AS profit_quartile
FROM s25
ORDER BY revenue DESC;


-- Q6. Promotion effectiveness: unit lift vs margin cost by category ------------
--     Lift is measured within the same store-product-month to remove seasonality bias.
WITH base AS (
    SELECT f.store_key, f.product_key, d.year, d.month,
           AVG(f.units_sold) FILTER (WHERE NOT pr.is_promo) AS base_units,
           AVG(f.units_sold) FILTER (WHERE pr.is_promo)     AS promo_units,
           SUM(f.profit)  FILTER (WHERE NOT pr.is_promo) / NULLIF(SUM(f.revenue) FILTER (WHERE NOT pr.is_promo), 0) AS base_margin,
           SUM(f.profit)  FILTER (WHERE pr.is_promo)     / NULLIF(SUM(f.revenue) FILTER (WHERE pr.is_promo), 0)     AS promo_margin
    FROM fact_sales f
    JOIN dim_date d       USING (date_key)
    JOIN dim_promotion pr USING (promo_key)
    GROUP BY f.store_key, f.product_key, d.year, d.month
)
SELECT p.category,
       ROUND(100 * (PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY promo_units / base_units) - 1)::numeric, 1) AS median_unit_lift_pct,
       ROUND(100 * AVG(base_margin)::numeric, 1)  AS margin_no_promo_pct,
       ROUND(100 * AVG(promo_margin)::numeric, 1) AS margin_on_promo_pct
FROM base b JOIN dim_product p USING (product_key)
WHERE base_units > 0 AND promo_units IS NOT NULL
GROUP BY p.category
ORDER BY median_unit_lift_pct DESC;


-- Q7. Festival impact: festive season (15 days pre-Diwali) vs rest of Sep-Nov --
SELECT p.category,
       ROUND(AVG(f.units_sold) FILTER (WHERE d.is_festive_season), 1)     AS avg_units_festive,
       ROUND(AVG(f.units_sold) FILTER (WHERE NOT d.is_festive_season), 1) AS avg_units_normal,
       ROUND(AVG(f.units_sold) FILTER (WHERE d.is_festive_season)
           / AVG(f.units_sold) FILTER (WHERE NOT d.is_festive_season), 2) AS festive_multiplier
FROM fact_sales f
JOIN dim_date d    USING (date_key)
JOIN dim_product p USING (product_key)
WHERE d.month IN (9, 10, 11)
GROUP BY p.category
ORDER BY festive_multiplier DESC;


-- Q8. Day-of-week profile by store format (index: 100 = that format's average) -
WITH dow AS (
    SELECT s.store_type, d.day_of_week, d.day_name, SUM(f.revenue) AS revenue
    FROM fact_sales f JOIN dim_date d USING (date_key) JOIN dim_store s USING (store_key)
    GROUP BY s.store_type, d.day_of_week, d.day_name
)
SELECT store_type, day_name,
       ROUND(100 * revenue / AVG(revenue) OVER (PARTITION BY store_type)) AS revenue_index
FROM dow
ORDER BY store_type, day_of_week;


-- Q9. 7-day and 28-day moving average of chain revenue (last 90 days) ----------
WITH daily AS (
    SELECT d.full_date, SUM(f.revenue) AS revenue
    FROM fact_sales f JOIN dim_date d USING (date_key)
    GROUP BY d.full_date
)
SELECT full_date,
       ROUND(revenue / 1e5, 2) AS revenue_lakh,
       ROUND(AVG(revenue) OVER (ORDER BY full_date ROWS BETWEEN 6  PRECEDING AND CURRENT ROW) / 1e5, 2) AS ma7_lakh,
       ROUND(AVG(revenue) OVER (ORDER BY full_date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) / 1e5, 2) AS ma28_lakh
FROM daily
ORDER BY full_date DESC
LIMIT 90;


-- Q10. ROLLUP: revenue by region > store type > category with subtotals (2025) -
SELECT COALESCE(s.region, '== ALL REGIONS ==')  AS region,
       COALESCE(s.store_type, '-- all types --') AS store_type,
       COALESCE(p.category, '(subtotal)')        AS category,
       ROUND(SUM(f.revenue) / 1e7, 2)            AS revenue_cr
FROM fact_sales f
JOIN dim_store s   USING (store_key)
JOIN dim_product p USING (product_key)
WHERE f.date_key BETWEEN 20250101 AND 20251231
GROUP BY ROLLUP (s.region, s.store_type, p.category)
ORDER BY s.region NULLS LAST, s.store_type NULLS LAST, p.category NULLS LAST;


-- Q11. Early warning: store-products whose 2025 units fell vs 2024 ------------
WITH yearly AS (
    SELECT f.store_key, f.product_key,
           SUM(f.units_sold) FILTER (WHERE f.date_key BETWEEN 20240101 AND 20241231) AS u2024,
           SUM(f.units_sold) FILTER (WHERE f.date_key BETWEEN 20250101 AND 20251231) AS u2025
    FROM fact_sales f
    WHERE f.date_key >= 20240101
    GROUP BY f.store_key, f.product_key
)
SELECT s.store_name, p.product_name, u2024, u2025,
       ROUND(100.0 * (u2025 - u2024) / u2024, 1) AS change_pct
FROM yearly y JOIN dim_store s USING (store_key) JOIN dim_product p USING (product_key)
WHERE u2025 < u2024
ORDER BY change_pct
LIMIT 15;


-- Q12. Discount leakage by category and Indian fiscal quarter -----------------
SELECT d.fiscal_year, d.fiscal_quarter, p.category,
       ROUND(SUM(f.discount_amount) / 1e5, 1)                     AS discount_lakh,
       ROUND(100 * SUM(f.discount_amount) / SUM(f.gross_sales), 2) AS discount_pct_of_gross
FROM fact_sales f JOIN dim_date d USING (date_key) JOIN dim_product p USING (product_key)
WHERE d.fiscal_year = 'FY2025-26'
GROUP BY d.fiscal_year, d.fiscal_quarter, p.category
ORDER BY d.fiscal_quarter, discount_pct_of_gross DESC;


-- Q13. Category share of each store's revenue (window over partition) --------
SELECT s.store_name, p.category,
       ROUND(100 * SUM(f.revenue) / SUM(SUM(f.revenue)) OVER (PARTITION BY s.store_name), 1) AS share_pct
FROM fact_sales f JOIN dim_store s USING (store_key) JOIN dim_product p USING (product_key)
WHERE f.date_key BETWEEN 20250101 AND 20251231
GROUP BY s.store_name, p.category
ORDER BY s.store_name, share_pct DESC;


-- Q14. Same-store sales growth by quarter (vs same quarter last year) --------
WITH q AS (
    SELECT s.store_name, d.year, d.quarter, SUM(f.revenue) AS revenue
    FROM fact_sales f JOIN dim_date d USING (date_key) JOIN dim_store s USING (store_key)
    GROUP BY s.store_name, d.year, d.quarter
)
SELECT store_name, year, quarter,
       ROUND(revenue / 1e7, 2) AS revenue_cr,
       ROUND(100 * (revenue / LAG(revenue, 4) OVER (PARTITION BY store_name ORDER BY year, quarter) - 1), 1) AS sssg_pct
FROM q
WHERE year = 2025
ORDER BY store_name, quarter;


-- Q15. Forecast accuracy by category from the ML write-back (run after train.py)
--      WAPE = sum|error| / sum(actual); accuracy = 1 - WAPE
SELECT p.category, fc.model_name,
       ROUND(100 * SUM(ABS(fc.forecast_units - fc.actual_units)) / SUM(fc.actual_units), 1)       AS wape_pct,
       ROUND(100 * (1 - SUM(ABS(fc.forecast_units - fc.actual_units)) / SUM(fc.actual_units)), 1) AS accuracy_pct,
       ROUND(100 * (SUM(fc.forecast_units) / SUM(fc.actual_units) - 1), 2)                       AS bias_pct
FROM fact_forecast fc JOIN dim_product p USING (product_key)
WHERE fc.actual_units IS NOT NULL
GROUP BY p.category, fc.model_name
ORDER BY accuracy_pct DESC;
