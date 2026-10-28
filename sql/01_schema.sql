-- =============================================================================
-- BharatMart Retail Data Warehouse : star schema (PostgreSQL 14+)
--
--                 dim_date
--                    |
--   dim_store --- fact_sales --- dim_product
--                    |
--              dim_promotion           fact_forecast (ML output, same grain)
--
-- Grain of fact_sales: one row per store x product x day.
-- Surrogate integer keys keep the fact table narrow and joins fast.
-- fact_sales is range-partitioned by date_key (one partition per year), so a
-- query filtered on 2025 only scans the 2025 partition (partition pruning).
-- =============================================================================

DROP TABLE IF EXISTS fact_forecast CASCADE;
DROP TABLE IF EXISTS fact_sales CASCADE;
DROP TABLE IF EXISTS dim_date CASCADE;
DROP TABLE IF EXISTS dim_store CASCADE;
DROP TABLE IF EXISTS dim_product CASCADE;
DROP TABLE IF EXISTS dim_promotion CASCADE;

-- ---------------------------------------------------------------- dimensions
CREATE TABLE dim_date (
    date_key          INTEGER PRIMARY KEY,          -- YYYYMMDD, e.g. 20251020
    full_date         DATE        NOT NULL UNIQUE,
    day_of_month      SMALLINT    NOT NULL,
    day_name          VARCHAR(9)  NOT NULL,
    day_of_week       SMALLINT    NOT NULL,          -- 1 = Monday ... 7 = Sunday (ISO)
    is_weekend        BOOLEAN     NOT NULL,
    week_of_year      SMALLINT    NOT NULL,
    month             SMALLINT    NOT NULL,
    month_name        VARCHAR(9)  NOT NULL,
    quarter           SMALLINT    NOT NULL,
    year              SMALLINT    NOT NULL,
    fiscal_year       VARCHAR(10) NOT NULL,          -- Indian FY (Apr-Mar), e.g. FY2025-26
    fiscal_quarter    VARCHAR(2)  NOT NULL,          -- Q1 = Apr-Jun
    is_holiday        BOOLEAN     NOT NULL,
    holiday_name      VARCHAR(50),
    days_to_diwali    SMALLINT    NOT NULL,          -- signed, clipped to [-30, 30]
    is_festive_season BOOLEAN     NOT NULL           -- 15 days before Diwali through Diwali
);

CREATE TABLE dim_store (
    store_key       SMALLSERIAL PRIMARY KEY,
    store_id        VARCHAR(5)   NOT NULL UNIQUE,    -- natural/business key
    store_name      VARCHAR(60)  NOT NULL,
    city            VARCHAR(30)  NOT NULL,
    state           VARCHAR(30)  NOT NULL,
    region          VARCHAR(10)  NOT NULL,
    store_type      VARCHAR(15)  NOT NULL CHECK (store_type IN ('Hypermarket', 'Supermarket', 'Express')),
    store_size_sqft INTEGER      NOT NULL CHECK (store_size_sqft > 0)
);

CREATE TABLE dim_product (
    product_key  SMALLSERIAL PRIMARY KEY,
    product_id   VARCHAR(5)  NOT NULL UNIQUE,
    product_name VARCHAR(60) NOT NULL,
    category     VARCHAR(30) NOT NULL,
    sub_category VARCHAR(30) NOT NULL,
    brand        VARCHAR(30) NOT NULL
);

CREATE TABLE dim_promotion (
    promo_key    SMALLINT PRIMARY KEY,
    discount_pct SMALLINT    NOT NULL UNIQUE CHECK (discount_pct BETWEEN 0 AND 90),
    is_promo     BOOLEAN     NOT NULL,
    promo_band   VARCHAR(20) NOT NULL                -- No Promo / Light / Medium / Deep
);

-- ---------------------------------------------------------------- facts
CREATE TABLE fact_sales (
    date_key        INTEGER       NOT NULL REFERENCES dim_date (date_key),
    store_key       SMALLINT      NOT NULL REFERENCES dim_store (store_key),
    product_key     SMALLINT      NOT NULL REFERENCES dim_product (product_key),
    promo_key       SMALLINT      NOT NULL REFERENCES dim_promotion (promo_key),
    units_sold      INTEGER       NOT NULL CHECK (units_sold >= 0),
    unit_price      NUMERIC(10,2) NOT NULL,
    unit_cost       NUMERIC(10,2) NOT NULL,
    gross_sales     NUMERIC(14,2) NOT NULL,
    discount_amount NUMERIC(14,2) NOT NULL,
    revenue         NUMERIC(14,2) NOT NULL,
    cost            NUMERIC(14,2) NOT NULL,
    profit          NUMERIC(14,2) NOT NULL,
    PRIMARY KEY (date_key, store_key, product_key)
) PARTITION BY RANGE (date_key);

CREATE TABLE fact_sales_2022 PARTITION OF fact_sales FOR VALUES FROM (20220101) TO (20230101);
CREATE TABLE fact_sales_2023 PARTITION OF fact_sales FOR VALUES FROM (20230101) TO (20240101);
CREATE TABLE fact_sales_2024 PARTITION OF fact_sales FOR VALUES FROM (20240101) TO (20250101);
CREATE TABLE fact_sales_2025 PARTITION OF fact_sales FOR VALUES FROM (20250101) TO (20260101);
CREATE TABLE fact_sales_2026 PARTITION OF fact_sales FOR VALUES FROM (20260101) TO (20270101);

-- Secondary indexes for the common slice-and-dice filters (created on every partition)
CREATE INDEX idx_fact_sales_store   ON fact_sales (store_key, date_key);
CREATE INDEX idx_fact_sales_product ON fact_sales (product_key, date_key);

-- ML forecasts written back into the warehouse so Power BI can show actual vs forecast
CREATE TABLE fact_forecast (
    date_key       INTEGER      NOT NULL REFERENCES dim_date (date_key),
    store_key      SMALLINT     NOT NULL REFERENCES dim_store (store_key),
    product_key    SMALLINT     NOT NULL REFERENCES dim_product (product_key),
    model_name     VARCHAR(30)  NOT NULL,
    forecast_units NUMERIC(10,2) NOT NULL,
    actual_units   INTEGER,                            -- NULL for true future dates
    created_at     TIMESTAMP    NOT NULL DEFAULT now(),
    PRIMARY KEY (date_key, store_key, product_key, model_name)
);
