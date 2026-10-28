-- =============================================================================
-- Reporting layer for Power BI.
-- Materialized views pre-aggregate the 584k-row fact table so dashboards stay
-- fast; refresh them after each ETL load (load_warehouse.py does this).
-- =============================================================================

DROP MATERIALIZED VIEW IF EXISTS mv_monthly_sales CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_daily_store_sales CASCADE;
DROP VIEW IF EXISTS vw_forecast_vs_actual CASCADE;

-- Monthly x store x product grain (~19k rows): main source for the dashboards
CREATE MATERIALIZED VIEW mv_monthly_sales AS
SELECT d.year,
       d.month,
       make_date(d.year, d.month, 1)             AS month_start,
       d.fiscal_year,
       s.store_id, s.store_name, s.city, s.region, s.store_type,
       p.product_id, p.product_name, p.category, p.sub_category, p.brand,
       SUM(f.units_sold)                          AS units_sold,
       SUM(f.gross_sales)                         AS gross_sales,
       SUM(f.discount_amount)                     AS discount_amount,
       SUM(f.revenue)                             AS revenue,
       SUM(f.cost)                                AS cost,
       SUM(f.profit)                              AS profit,
       SUM(CASE WHEN pr.is_promo THEN f.revenue ELSE 0 END) AS promo_revenue
FROM fact_sales f
JOIN dim_date d       ON d.date_key = f.date_key
JOIN dim_store s      ON s.store_key = f.store_key
JOIN dim_product p    ON p.product_key = f.product_key
JOIN dim_promotion pr ON pr.promo_key = f.promo_key
GROUP BY d.year, d.month, d.fiscal_year, s.store_id, s.store_name, s.city, s.region, s.store_type,
         p.product_id, p.product_name, p.category, p.sub_category, p.brand;

CREATE INDEX ON mv_monthly_sales (month_start);

-- Daily x store x category grain: for daily trend lines and the festival page
CREATE MATERIALIZED VIEW mv_daily_store_sales AS
SELECT d.full_date, d.day_name, d.is_weekend, d.is_holiday, d.holiday_name, d.days_to_diwali,
       s.store_id, s.store_name, s.region, s.store_type,
       p.category,
       SUM(f.units_sold) AS units_sold,
       SUM(f.revenue)    AS revenue,
       SUM(f.profit)     AS profit
FROM fact_sales f
JOIN dim_date d    ON d.date_key = f.date_key
JOIN dim_store s   ON s.store_key = f.store_key
JOIN dim_product p ON p.product_key = f.product_key
GROUP BY d.full_date, d.day_name, d.is_weekend, d.is_holiday, d.holiday_name, d.days_to_diwali,
         s.store_id, s.store_name, s.region, s.store_type, p.category;

CREATE INDEX ON mv_daily_store_sales (full_date);

-- Forecast vs actual for the ML page of the dashboard
CREATE VIEW vw_forecast_vs_actual AS
SELECT d.full_date, s.store_id, s.store_name, s.region,
       p.product_id, p.product_name, p.category,
       fc.model_name, fc.forecast_units, fc.actual_units,
       fc.forecast_units - fc.actual_units        AS error_units,
       ABS(fc.forecast_units - fc.actual_units)   AS abs_error_units
FROM fact_forecast fc
JOIN dim_date d    ON d.date_key = fc.date_key
JOIN dim_store s   ON s.store_key = fc.store_key
JOIN dim_product p ON p.product_key = fc.product_key;

-- Flat fact keyed by natural keys (date / store_id / product_id). Same columns as
-- data/processed/retail_sales_clean.csv, so the Power BI model and DAX are identical
-- whether you connect to PostgreSQL or import the CSVs.
DROP VIEW IF EXISTS vw_sales_fact CASCADE;
CREATE VIEW vw_sales_fact AS
SELECT d.full_date AS date, s.store_id, p.product_id,
       pr.discount_pct, pr.is_promo::int AS is_promo, pr.promo_band,
       f.units_sold, f.unit_price, f.unit_cost, f.gross_sales, f.discount_amount,
       f.revenue, f.cost, f.profit
FROM fact_sales f
JOIN dim_date d       ON d.date_key = f.date_key
JOIN dim_store s      ON s.store_key = f.store_key
JOIN dim_product p    ON p.product_key = f.product_key
JOIN dim_promotion pr ON pr.promo_key = f.promo_key;
