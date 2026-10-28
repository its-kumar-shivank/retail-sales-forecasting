# 🗄️ Phase 4: SQL & Data Warehousing

> **Welcome to Phase 4!** In [Phase 3: Exploratory Data Analysis](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/learning/03-exploratory-data-analysis/README.md), we explored BharatMart's retail trends using Python and charts. But in real-world retail companies, data doesn't sit in lone CSV files on a laptop. Millions of daily sales transactions flow into a central **data warehouse**. In this phase, we explore how BharatMart stores, models, and queries 584,000 sales records using **PostgreSQL** and **SQL**.

---

## 📋 Table of Contents

1. [What is a Database?](#1-what-is-a-database)
2. [SQL Basics: Talking to the Database](#2-sql-basics-talking-to-the-database)
3. [Star Schema: How BharatMart Organizes Data](#3-star-schema-how-bharatmart-organizes-data)
4. [Surrogate Keys: The Unsung Heroes of Data Warehousing](#4-surrogate-keys-the-unsung-heroes-of-data-warehousing)
5. [Table Partitioning: Dividing and Conquering Big Data](#5-table-partitioning-dividing-and-conquering-big-data)
6. [Window Functions: The Superpower of SQL](#6-window-functions-the-superpower-of-sql)
7. [The 15 Business Queries: Answering Real Retail Questions](#7-the-15-business-queries-answering-real-retail-questions)
8. [Materialized Views: Turbocharging Dashboards](#8-materialized-views-turbocharging-dashboards)
9. [Loading Data into the Warehouse: Fast ETL with COPY](#9-loading-data-into-the-warehouse-fast-etl-with-copy)
10. [Key Takeaways](#10-key-takeaways)
11. [Next Steps](#next-steps)

---

## 1. What is a Database?

### 🗄️ The Filing Cabinet Analogy

Imagine you own a single neighborhood grocery shop. You keep receipts in a cardboard box on your desk. When you need to check yesterday's sales, you rifle through the box. That cardboard box is like a **CSV file** — simple, portable, and fine for small amounts of paperwork.

Now imagine BharatMart expands to **10 massive hypermarkets across India**, handling **584,000 transactions** across 4 years.
- What if 5 store managers try to update inventory at the exact same second?
- What if someone accidentally types `"Fifty"` instead of `50` for units sold?
- What if looking up a customer's order requires flipping through 600,000 paper sheets?

A cardboard box collapses under that weight. You need a **modern electronic filing cabinet**: fireproof, organized with labeled folders, guarded by a security system, and capable of answering questions in milliseconds. That electronic filing cabinet is a **database**.

```
┌─────────────────────────────────────────────────────────────┐
│                      DATABASE SYSTEM                        │
│                                                             │
│   ┌───────────────────┐               ┌───────────────────┐ │
│   │    dim_store      │               │   dim_product     │ │
│   │  (Store Directory)│               │ (Product Catalog) │ │
│   └─────────┬─────────┘               └─────────┬─────────┘ │
│             │                                   │           │
│             └───────────────┬───────────────────┘           │
│                             ▼                               │
│                   ┌───────────────────┐                     │
│                   │    fact_sales     │                     │
│                   │ (584,000 Records) │                     │
│                   └───────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### 🛑 Why Not Just Use CSV Files?

While CSV (Comma-Separated Values) files are great for sharing small tables, they fail at enterprise scale:

| Capability | CSV Files | Relational Database (PostgreSQL) |
| :--- | :--- | :--- |
| **Speed** | Reads the whole file from disk into RAM every time | Uses indexed B-Trees to find 1 row out of 1 million in < 1 ms |
| **Concurrency** | If two people write at once, the file gets corrupted | Thousands of users can read and write simultaneously |
| **Data Integrity** | Accepts invalid data (e.g. negative prices, text in dates) | Strict data types and `CHECK` constraints prevent garbage |
| **Querying Power** | Requires writing custom Python loops or Pandas scripts | Expressive SQL language with joins, aggregations, and filters |
| **Security** | Anyone with file access sees everything | Role-based permissions down to specific columns or rows |

### 🐘 PostgreSQL: The Project Database

In this project, we use **PostgreSQL** (often called "Postgres"), the world's most advanced open-source relational database. Postgres is rock-solid, supports advanced analytical queries, and easily handles millions of rows.

Inside PostgreSQL:
- A **Database** is your company's entire warehouse (`retail_dw`).
- A **Table** is like an individual spreadsheet tab (with named columns and structured rows).
- A **Row** (or record) is a single observation (e.g., store S001 selling product P001 on Oct 24, 2025).
- A **Column** is a specific attribute with a defined data type (`INTEGER`, `DATE`, `NUMERIC`).

---

## 2. SQL Basics: Talking to the Database

**SQL** stands for **Structured Query Language** (pronounced *"sequel"* or *"S-Q-L"*). It is the universal language used by data analysts, data engineers, and machine learning engineers to communicate with databases.

Unlike Python, where you write step-by-step instructions (*how* to compute something), SQL is **declarative** (*what* data you want).

Here are the six fundamental clauses you need to know:

### 1. `SELECT`: Which columns do you want?
Think of `SELECT` as picking which columns to keep from a spreadsheet.
```sql
SELECT product_name, category, unit_price
```

### 2. `FROM`: Which table should we look in?
Tells the database where the table is located.
```sql
SELECT product_name, category
FROM dim_product;
```

### 3. `WHERE`: How should we filter rows?
Like clicking the "Filter" dropdown in Excel to hide rows that do not match your condition.
```sql
SELECT product_name, unit_price
FROM dim_product
WHERE category = 'Grocery' AND unit_price < 200.00;
```

### 4. `GROUP BY`: How should we aggregate data?
Just like an Excel **Pivot Table**. It groups rows sharing the same value and calculates summary statistics (such as `SUM()`, `AVG()`, `COUNT()`, `MIN()`, or `MAX()`).
```sql
SELECT category, COUNT(*) AS item_count, ROUND(AVG(unit_price), 2) AS avg_price
FROM dim_product
GROUP BY category;
```

### 5. `ORDER BY`: How should the results be sorted?
Sorts results in ascending (`ASC`, default) or descending (`DESC`) order.
```sql
SELECT product_name, unit_price
FROM dim_product
ORDER BY unit_price DESC;
```

### 6. `JOIN`: How do we connect two tables?
Real-world data is split across tables to prevent duplication. A `JOIN` matches rows from two tables using a common identifier.
```sql
-- Combine sales with product names using product_key
SELECT f.date_key, p.product_name, f.units_sold, f.revenue
FROM fact_sales f
JOIN dim_product p ON f.product_key = p.product_key
WHERE f.date_key = 20251020;
```

> [!TIP]
> **SQL Order of Execution**: SQL queries are written starting with `SELECT`, but the database actually executes them in this order:
> 1. `FROM` & `JOIN` (find tables) ➔ 2. `WHERE` (filter rows) ➔ 3. `GROUP BY` (aggregate) ➔ 4. `HAVING` (filter aggregates) ➔ 5. `SELECT` (pick columns) ➔ 6. `ORDER BY` (sort) ➔ 7. `LIMIT` (truncate).

---

## 3. Star Schema: How BharatMart Organizes Data

Data warehousing uses a battle-tested design pattern called the **Star Schema** (pioneered by Ralph Kimball). 

You can view the full database definition in [`sql/01_schema.sql`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/01_schema.sql).

### 🌟 Why is it Called a "Star"?

If you draw lines between the tables, the diagram resembles a star: one large table in the center surrounded by smaller lookup tables at the points.

```
                     ┌──────────────────────────────┐
                     │           dim_date           │
                     │ ──────────────────────────── │
                     │ date_key (PK)                │
                     │ full_date, month, year       │
                     │ is_holiday, is_festive_season│
                     └──────────────┬───────────────┘
                                    │
                                    │
┌──────────────────────────────┐    │    ┌──────────────────────────────┐
│          dim_store           │    │    │         dim_product          │
│ ──────────────────────────── │    ▼    │ ──────────────────────────── │
│ store_key (PK)               ├───►*◄───┤ product_key (PK)             │
│ store_id ('S001')            │ fact_   │ product_id ('P001')          │
│ store_name, city, region     │ sales   │ product_name, category, brand│
│ store_type, store_size_sqft  │         └──────────────────────────────┘
└──────────────────────────────┘    ▲
                                    │
                                    │
                     ┌──────────────┴───────────────┐
                     │        dim_promotion         │
                     │ ──────────────────────────── │
                     │ promo_key (PK)               │
                     │ discount_pct, is_promo       │
                     │ promo_band ('Deep 30%+')     │
                     └──────────────────────────────┘
```

The database is divided into two distinct types of tables:

### 1. The Fact Table (Center): `fact_sales`
The **fact table** records measurements of business events. Every time a customer buys something, a fact is created.
- **Grain** (the meaning of one single row): **One store selling one product on one day**.
- **Contains mostly numbers (metrics)**:
  - `units_sold`: how many items left the shelf (e.g. 42)
  - `unit_price`, `unit_cost`: price and cost per item (e.g. ₹120.00, ₹95.00)
  - `gross_sales`: units × regular price
  - `discount_amount`: revenue forgone due to promotions
  - `revenue`: actual cash collected after discount
  - `cost`: total cost of goods sold
  - `profit`: `revenue - cost`
- **Keys**: Small integer references (`date_key`, `store_key`, `product_key`, `promo_key`) pointing to the surrounding dimensions.

### 2. Dimension Tables (Points of the Star): Context and Filters
Dimension tables provide the **"who, what, where, and when"** context for every metric. They contain descriptive text used for slicing, dicing, and grouping:

1. [`dim_date`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/01_schema.sql#L24-L42):
   - Every single calendar date from 2022 to 2026.
   - Rich retail calendar attributes: day of week, Indian fiscal quarter (`FY2025-26 Q3`), holiday indicators, and days countdown to Diwali (`days_to_diwali`).
2. [`dim_store`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/01_schema.sql#L44-L53):
   - Store metadata: name (`"Indiranagar Supermarket"`), city, state, region (`South`), store format (`Hypermarket`), and floor area (`store_size_sqft`).
3. [`dim_product`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/01_schema.sql#L55-L63):
   - Product hierarchy: name (`"Basmati Rice 5kg"`), category (`Grocery`), sub-category (`Grains`), and brand.
4. [`dim_promotion`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/01_schema.sql#L64-L70):
   - Promotional discount bands: `No Promo (0%)`, `Light (10-15%)`, `Medium (20-25%)`, `Deep (30%+)`.

### 💡 Why Use a Star Schema?

1. **Eliminates Repetition (Storage Efficiency)**: Instead of repeating the string `"Indiranagar Supermarket, Bangalore, Karnataka, South"` on all 58,400 sales records for that store, we store that text **once** in `dim_store` and just store a 2-byte integer (`store_key = 3`) in `fact_sales`.
2. **Blazing Fast Queries**: Joining integer keys is orders of magnitude faster for databases than comparing long strings of text.
3. **Intuitive for Business**: Any analyst can easily understand the model: facts are what happened; dimensions are how you slice it.

---

## 4. Surrogate Keys: The Unsung Heroes of Data Warehousing

Take a look at `dim_store`:
```sql
CREATE TABLE dim_store (
    store_key       SMALLSERIAL PRIMARY KEY,  -- Surrogate key (1, 2, 3...)
    store_id        VARCHAR(5)   NOT NULL UNIQUE, -- Natural key ('S001')
    store_name      VARCHAR(60)  NOT NULL,
    ...
);
```

Notice there are two IDs for the same store: `store_id` and `store_key`. What is the difference?

### Natural Key vs. Surrogate Key

- **Natural Key (Business Key)**: The identifier used in the real world (e.g. Store ID `'S001'`, Product ID `'P042'`, or PAN/Aadhaar card number). It is usually text and defined by external software (like POS checkout registers).
- **Surrogate Key**: An artificial, meaningless integer generated purely inside the data warehouse (e.g. `1`, `2`, `3`).

```
                ┌────────────────────────────────────────┐
                │          dim_store Table               │
                ├───────────┬──────────┬─────────────────┤
Surrogate Key ─►│ store_key │ store_id │ store_name      │◄─ Natural Key
(Integer: 1, 2) │     1     │  'S001'  │ Delhi Saket     │   ('S001')
                │     2     │  'S002'  │ Mumbai Andheri  │
                └───────────┴──────────┴─────────────────┘
```

### Why Use Surrogate Keys Instead of Natural Keys?

1. **Speed & Storage**: 
   - Comparing numbers (like `3 == 3`) takes **1 CPU clock cycle**.
   - Comparing strings (like `'S003' == 'S003'`) requires inspecting each byte individually.
   - Storing a 2-byte integer (`SMALLINT`) across 584,000 rows saves megabytes of disk and RAM compared to variable-length text strings.
2. **Protection Against Business Changes**: 
   - What if BharatMart rebrands or renumbers stores (e.g. `'S001'` becomes `'DEL-01'`)? 
   - With natural keys, you would have to update 58,400 rows in the massive fact table.
   - With surrogate keys, the fact table uses `store_key = 1`, which **never changes**! You only update a single row in `dim_store`.
3. **Handling History (Slowly Changing Dimensions)**: 
   - If store S001 remodels from a 15,000 sq ft Supermarket to a 50,000 sq ft Hypermarket, surrogate keys allow the warehouse to assign key `1` to the old size and key `11` to the new size, preserving historical accuracy.

---

## 5. Table Partitioning: Dividing and Conquering Big Data

### 📖 The Phone Book Analogy

Imagine a telephone directory for all of India with 100 million numbers bound into one massive 50,000-page book. If you want to look up a phone number from the year 2025, flipping through that gigantic book takes forever.

Instead, what if the publisher split it into **5 slim annual volumes**:
- Book 1: 2022
- Book 2: 2023
- Book 3: 2024
- Book 4: 2025
- Book 5: 2026

If you want numbers for 2025, you don't even open Books 1, 2, 3, or 5. You reach directly for the 2025 book!

### How BharatMart Uses Partitioning

In [`sql/01_schema.sql`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/01_schema.sql#L72-L93), `fact_sales` is range-partitioned by `date_key`:

```sql
CREATE TABLE fact_sales (
    date_key    INTEGER       NOT NULL REFERENCES dim_date (date_key),
    store_key   SMALLINT      NOT NULL REFERENCES dim_store (store_key),
    ...
) PARTITION BY RANGE (date_key);

-- Child tables physically stored in separate files on disk:
CREATE TABLE fact_sales_2022 PARTITION OF fact_sales FOR VALUES FROM (20220101) TO (20230101);
CREATE TABLE fact_sales_2023 PARTITION OF fact_sales FOR VALUES FROM (20230101) TO (20240101);
CREATE TABLE fact_sales_2024 PARTITION OF fact_sales FOR VALUES FROM (20240101) TO (20250101);
CREATE TABLE fact_sales_2025 PARTITION OF fact_sales FOR VALUES FROM (20250101) TO (20260101);
CREATE TABLE fact_sales_2026 PARTITION OF fact_sales FOR VALUES FROM (20260101) TO (20270101);
```

### ⚡ Partition Pruning

When you run a query like:
```sql
SELECT SUM(revenue) 
FROM fact_sales 
WHERE date_key BETWEEN 20250101 AND 20251231;
```

PostgreSQL's query planner recognizes that only `fact_sales_2025` can possibly satisfy this condition. It completely **ignores** (prunes) the 2022, 2023, 2024, and 2026 tables!

```
Incoming Query: WHERE date_key BETWEEN 20250101 AND 20251231
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [2022, 2023, 2024, 2026]              [fact_sales_2025]
      🚫 SKIPPED COMPLETELY                 ✅ SCANNED DIRECTLY
      (0 disk reads!)                       (100% of target data)
```

**Result**: Queries run up to **5x faster** and consume a fraction of memory.

---

## 6. Window Functions: The Superpower of SQL

If you only learn one advanced SQL concept, make it **Window Functions**. They are among the most frequently asked topics in data science and data engineering interviews.

### Group By vs. Window Functions

- **`GROUP BY` collapses rows**: If you start with 10 rows and group by region, you get back 4 rows (one per region). The individual row details are lost.
- **Window Functions calculate without collapsing**: They compute an aggregate or ranking across a "window" of related rows, but **keep every single original row intact**!

```
GROUP BY (Collapses)                WINDOW FUNCTION (Preserves Rows)
Row 1 (North, ₹100) ┐               Row 1 (North, ₹100) ──► ₹100, Total: ₹350
Row 2 (North, ₹250) ┴─► North: ₹350  Row 2 (North, ₹250) ──► ₹250, Total: ₹350
Row 3 (South, ₹400) ──► South: ₹400  Row 3 (South, ₹400) ──► ₹400, Total: ₹400
```

Every window function uses the `OVER (...)` clause, which can take two optional arguments:
- `PARTITION BY`: Divides the rows into groups (like separate rooms).
- `ORDER BY`: Orders the rows inside each group (like standing in line).

Here are the 5 essential window functions used in this project:

---

### 1. `ROW_NUMBER()`: Assigning Sequential Row Numbers
Numbers rows sequentially starting from 1.
```sql
SELECT store_name, region, store_size_sqft,
       ROW_NUMBER() OVER (PARTITION BY region ORDER BY store_size_sqft DESC) AS size_rank
FROM dim_store;
```
*Result: Inside each region, the largest store gets 1, the next gets 2, etc.*

---

### 2. `RANK()` vs. `DENSE_RANK()`: Handling Ties
What happens if two products have identical sales?
- `RANK()` leaves gaps after ties: `1, 2, 2, 4` (notice 3 was skipped!).
- `DENSE_RANK()` does not leave gaps: `1, 2, 2, 3`.

```sql
SELECT product_name, category, revenue,
       DENSE_RANK() OVER (PARTITION BY category ORDER BY revenue DESC) AS category_rank
FROM product_summary;
```

---

### 3. `LAG()` and `LEAD()`: Time Travel
- `LAG(column, N)`: Peeks at the value from **N rows before** the current row.
- `LEAD(column, N)`: Peeks at the value from **N rows ahead** of the current row.

This is the standard technique for computing **growth rates**:
```sql
-- Compare this month's revenue to last month's revenue
SELECT month, revenue,
       LAG(revenue, 1) OVER (ORDER BY month) AS prev_month_revenue,
       ROUND(100.0 * (revenue - LAG(revenue, 1) OVER (ORDER BY month)) 
             / LAG(revenue, 1) OVER (ORDER BY month), 1) AS mom_growth_pct
FROM monthly_sales;
```

---

### 4. Running Totals: `SUM() OVER (...)`
Compute a running cumulative sum across time without collapsing rows:
```sql
SELECT month, revenue,
       -- Running sum that accumulates month after month:
       SUM(revenue) OVER (ORDER BY month) AS running_total_revenue
FROM monthly_sales;
```

---

### 5. Moving Averages: `AVG() OVER (ROWS BETWEEN ...)`
In retail, daily sales bounce wildly between weekends and weekdays. A **7-day moving average** smooths out that noise by averaging the current day and the previous 6 days:

```sql
SELECT full_date, revenue,
       AVG(revenue) OVER (
           ORDER BY full_date 
           ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
       ) AS moving_avg_7d
FROM daily_sales;
```

---

## 7. The 15 Business Queries: Answering Real Retail Questions

All analytical queries are located in [`sql/02_analytics_queries.sql`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/02_analytics_queries.sql). These are not toy queries — they represent genuine executive questions asked by retail directors at companies like Walmart, Reliance Retail, or DMart.

Here are 5 of the most important queries and how they work:

---

### 📈 Q1: Month-over-Month (MoM) & Year-over-Year (YoY) Growth
*Business Question: Is BharatMart growing compared to last month? How about compared to the same month last year?*

[`sql/02_analytics_queries.sql#L9-L21`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/02_analytics_queries.sql#L9-L21):
```sql
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
```

**How it works:**
1. The Common Table Expression (CTE) `monthly` aggregates daily sales into total revenue per month.
2. `LAG(revenue, 1)` pulls the previous month's revenue to compute MoM % change.
3. `LAG(revenue, 12)` looks back exactly 12 months to compute YoY % change, neutralizing normal seasonal swings.
4. `revenue / 1e7` converts values into Indian Crores (1 Crore = 10,000,000).

---

### 🏆 Q3: Top-3 Products by Revenue in Each Region
*Business Question: Which products are the local champions in the North, South, East, and West?*

[`sql/02_analytics_queries.sql#L32-L46`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/02_analytics_queries.sql#L32-L46):
```sql
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
FROM ranked 
WHERE rnk <= 3
ORDER BY region, rnk;
```

**How it works:**
1. `WHERE f.date_key BETWEEN ...` triggers **partition pruning**, scanning only the 2025 partition table.
2. `DENSE_RANK() OVER (PARTITION BY s.region ORDER BY SUM(f.revenue) DESC)` ranks products from #1 onwards separately inside each geographic region.
3. The outer query filters `WHERE rnk <= 3` to keep only the top 3 items per region.

---

### ⚖️ Q4: Pareto 80/20 Analysis (ABC Classification)
*Business Question: The Pareto Principle states that ~80% of revenue comes from ~20% of products. Which products are our "Class A" essentials?*

[`sql/02_analytics_queries.sql#L48-L66`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/02_analytics_queries.sql#L48-L66):
```sql
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
```

**How it works:**
1. Products are sorted in descending order of revenue.
2. `SUM(revenue) OVER (ORDER BY revenue DESC)` calculates the running cumulative revenue.
3. `SUM(revenue) OVER ()` (with empty brackets) computes the total grand sum of all products across the entire chain.
4. Dividing running revenue by total revenue yields `cum_share` (0.0 to 1.0).
5. A `CASE` statement tags products as:
   - **A**: Top contributors driving the first 80% of sales (never let these run out of stock!).
   - **B**: Steady performers driving the next 15% (80% to 95%).
   - **C**: Long-tail slow movers driving the bottom 5%.

---

### 🏢 Q5: Store Scorecard & Space Productivity
*Business Question: Big stores have higher revenue, but which store makes the most money per square foot of real estate?*

[`sql/02_analytics_queries.sql#L68-L85`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/02_analytics_queries.sql#L68-L85):
```sql
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
```

**How it works:**
- Calculates **Revenue per Square Foot** (`revenue / store_size_sqft`), the gold standard KPI for physical retail efficiency.
- `NTILE(4) OVER (ORDER BY profit DESC)` divides the stores into 4 equal quartiles (Quartile 1 = top 25% profit earners).
- An Express store might have lower total revenue than a Hypermarket, but rank #1 in `productivity_rank` due to efficient floor utilization.

---

### 🏷️ Q6: Promotion Effectiveness: Volume Lift vs. Margin Cost
*Business Question: When we offer discounts, do sales increase enough to make up for the profit margin we gave away?*

[`sql/02_analytics_queries.sql#L87-L108`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/02_analytics_queries.sql#L87-L108):
```sql
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
```

**How it works:**
- Uses modern PostgreSQL `FILTER (WHERE ...)` conditional aggregation clauses.
- Measures baseline vs promo performance within the **exact same store, product, and month** — eliminating seasonal bias.
- Uses `PERCENTILE_CONT(0.5)` to calculate the **median** lift, ensuring single-day outliers don't skew promotional decisions.

---

## 8. Materialized Views: Turbocharging Dashboards

When connecting business intelligence tools like **Power BI** or **Streamlit** to a database, you cannot have the dashboard run expensive queries across 584,000 rows every time a user clicks a button. The dashboard would feel sluggish and freeze.

To solve this, we use **Materialized Views**, defined in [`sql/03_bi_views.sql`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/03_bi_views.sql).

### Standard View vs. Materialized View

```
┌────────────────────────────────────────────────────────────────────────┐
│ STANDARD VIEW: Like a Browser Bookmark                                 │
│ - Saves the query SQL text.                                            │
│ - When you run SELECT * FROM my_view, it re-executes the query from    │
│   scratch on the underlying tables.                                    │
│ - Fresh data every time, but slow for heavy calculations.              │
├────────────────────────────────────────────────────────────────────────┤
│ MATERIALIZED VIEW: Like Printing the Webpage to Paper                  │
│ - Saves the query SQL text AND executes it once, storing the computed  │
│   results physically on disk.                                          │
│ - Instant reading speed!                                               │
│ - Requires a quick REFRESH when new sales data arrives.                │
└────────────────────────────────────────────────────────────────────────┘
```

### The Two Materialized Views in BharatMart

1. [`mv_monthly_sales`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/03_bi_views.sql#L12-L35):
   - Pre-aggregates daily sales to **Month × Store × Product** grain.
   - Shrinks **584,000 rows down to ~19,000 rows** (a 97% reduction!).
   - Feeds executive monthly trend charts in < 5 milliseconds.

2. [`mv_daily_store_sales`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/03_bi_views.sql#L37-L52):
   - Aggregates sales to **Day × Store × Category** grain.
   - Includes calendar flags (`is_holiday`, `days_to_diwali`) for festival trendlines and category comparisons.

```sql
-- Refreshing after ETL takes less than 1 second:
REFRESH MATERIALIZED VIEW mv_monthly_sales;
REFRESH MATERIALIZED VIEW mv_daily_store_sales;
```

---

## 9. Loading Data into the Warehouse: Fast ETL with COPY

How do 584,000 rows get from our cleaned CSV into PostgreSQL? That is the job of [`src/load_warehouse.py`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/src/load_warehouse.py).

### 🐢 `INSERT` (Slow) vs. ⚡ `COPY` (Fast)

Most beginners load data by writing a loop that executes `INSERT` statements one by one:

```python
# The slow way: 584,000 network round-trips!
for row in records:
    cursor.execute("INSERT INTO fact_sales VALUES (%s, %s, ...)", row)
```

Why is this slow? For every row, Python sends a packet over the network, Postgres parses the SQL statement, creates an execution plan, acquires a lock, writes the row, and sends back an acknowledgment. Doing that 584,000 times takes **15 to 30 minutes**!

Instead, [`src/load_warehouse.py`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/src/load_warehouse.py#L48-L55) uses PostgreSQL's native **`COPY FROM STDIN`** protocol:

```python
def copy_df(conn, df: pd.DataFrame, table: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(f"COPY {table} ({','.join(df.columns)}) FROM STDIN WITH (FORMAT csv)", buf)
    conn.commit()
```

- Converts the Pandas DataFrame into an in-memory byte stream (`io.StringIO`).
- Streams the entire table as a single bulk binary transfer directly into the database engine.
- Loads all **584,000 records in under 3 seconds**!

### 🔍 Data Reconciliation: Trust, but Verify

In corporate data engineering, the ETL (Extract, Transform, Load) script is never considered complete until automated **reconciliation checks** prove zero data loss.

In [`src/load_warehouse.py`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/src/load_warehouse.py#L155-L163), the script queries the newly loaded database and compares the totals against the source CSV file:

```python
with conn.cursor() as cur:
    cur.execute("SELECT COUNT(*), SUM(units_sold), ROUND(SUM(revenue), 2) FROM fact_sales")
    n, units, rev = cur.fetchone()

# 1. Did we lose any rows?
assert n == len(df), f"Row count mismatch {n} vs {len(df)}"

# 2. Did we lose any physical products sold?
assert units == df.units_sold.sum(), "Units sold mismatch"

# 3. Does the total revenue match down to the exact rupee?
assert abs(float(rev) - df.revenue.sum()) < 1.0, "Revenue mismatch"

print(f"Validated: {n:,} rows, {units:,} units, ₹{float(rev):,.2f} revenue — matches source.")
```

If even one rupee or one unit is unaccounted for, the pipeline halts immediately and alerts the engineer.

---

## 10. Key Takeaways

1. **Databases Provide Structure & Speed**: While CSV files are good for simple exchanges, relational databases like PostgreSQL enforce data integrity, provide multi-user concurrency, and query millions of rows in milliseconds.
2. **The Star Schema Keeps Things Clean**: Splitting data into a central numeric **fact table** (`fact_sales`) and descriptive **dimension tables** (`dim_date`, `dim_store`, `dim_product`, `dim_promotion`) avoids duplication and makes business reporting easy.
3. **Surrogate Keys Improve Performance**: Small integer keys (`1, 2, 3`) speed up database joins and shield the warehouse from changes in external business codes.
4. **Partitioning Slices Big Tables**: Dividing `fact_sales` into yearly partitions allows PostgreSQL to prune irrelevant years, speeding up queries by 5x.
5. **Window Functions Perform Magic**: Functions like `LAG()`, `DENSE_RANK()`, and `AVG() OVER (ROWS BETWEEN ...)` calculate running metrics, rankings, and moving averages without collapsing original rows.
6. **Materialized Views Power Dashboards**: Pre-computing monthly and daily aggregates reduces 584,000 rows to 19,000 rows, allowing BI tools like Power BI to load instantly.
7. **Bulk Loading via `COPY`**: Streaming data with `COPY FROM STDIN` is hundreds of times faster than individual `INSERT` statements.

---

## Next Steps

Now that our data is cleanly structured and stored in the data warehouse, we are ready to build predictive features for machine learning!

👉 Proceed to [Phase 5: Feature Engineering →](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/learning/05-feature-engineering/README.md)
