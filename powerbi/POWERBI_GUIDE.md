# Power BI Dashboard Guide

A `.pbix` file can only be created inside Power BI Desktop, so this folder gives you everything needed to build the dashboard in about 45–60 minutes: the data model, every DAX measure (`measures.dax`), a colour theme (`bharatmart_theme.json`), and a page-by-page layout.

> **On a Mac?** Power BI Desktop is Windows-only. Options: a college lab / friend's Windows PC, a Windows virtual machine (e.g. Parallels or UTM), or the Power BI web service (app.powerbi.com), which can build reports from uploaded CSV files, though its modelling features are more limited than Desktop. The Streamlit app in this project covers the same insights if you can't get Power BI at all.

---

## 1. Load the data (pick one route)

### Route A: connect to PostgreSQL (best for the resume story)
1. Run `python src/load_warehouse.py` so the warehouse exists.
2. Power BI Desktop → **Get Data → PostgreSQL database** → Server `localhost`, Database `retail_dw`, mode **Import**.
3. Select: `vw_sales_fact`, `dim_date`, `dim_store`, `dim_product`, `vw_forecast_vs_actual`.
4. If Power BI asks for a driver, install the Npgsql provider it links to, then restart Power BI.

### Route B: import CSV files (no database needed)
**Get Data → Text/CSV** and load:
- `data/processed/retail_sales_clean.csv`
- `data/exports/dim_date.csv`, `dim_store.csv`, `dim_product.csv`
- `data/exports/vw_forecast_vs_actual.csv`

### Both routes: also load
`data/exports/future_forecast_28d.csv` (next 28 days forecast).

### Rename tables (Model view → double-click table name)
| Source | Rename to |
|---|---|
| `vw_sales_fact` or `retail_sales_clean` | `Sales` |
| `vw_forecast_vs_actual` | `ForecastEval` |
| `future_forecast_28d` | `FutureForecast` |

In Power Query, make sure date columns are type **Date** and `store_id` / `product_id` are **Text**.

## 2. Build the star-schema model

In **Model view**, create these relationships (all *many-to-one, single direction*, dimension → fact):

| From (many) | To (one) |
|---|---|
| `Sales[date]` | `dim_date[full_date]` |
| `Sales[store_id]` | `dim_store[store_id]` |
| `Sales[product_id]` | `dim_product[product_id]` |
| `ForecastEval[full_date]` | `dim_date[full_date]` |
| `ForecastEval[store_id]` | `dim_store[store_id]` |
| `ForecastEval[product_id]` | `dim_product[product_id]` |
| `FutureForecast[date]` | `dim_date[full_date]` |
| `FutureForecast[store_id]` | `dim_store[store_id]` |
| `FutureForecast[product_id]` | `dim_product[product_id]` |

Then:
- Select `dim_date` → **Table tools → Mark as date table** → `full_date`. Time-intelligence measures need this.
- `dim_date[month_name]` → **Sort by column** → `month`; `dim_date[day_name]` → sort by `day_of_week`.
- Hide the key columns in the fact tables (right-click → Hide in report view) so users slice by dimensions only.
- In Route B, `retail_sales_clean` also contains store/product names. Hide them too and always use the dimension columns, so every visual filters all fact tables consistently.

## 3. Add the measures
**Home → Enter data** → name the table `_Measures` → Load. Then for each measure in `measures.dax`: select `_Measures` → **New measure** → paste one definition. Set formats: `%` measures as Percentage (1 decimal), revenue as Currency (₹, 0 decimals).

## 4. Apply the theme
**View → Themes → Browse for themes** → `bharatmart_theme.json`.

## 5. Report pages

Put a slicer panel on every page: `dim_date[fiscal_year]`, `dim_store[region]`, `dim_store[store_type]`, `dim_product[category]`. Use **View → Sync slicers** so selections carry across pages.

### Page 1: Executive Overview
- **KPI cards**: Total Revenue, Revenue YoY %, Profit Margin %, Total Units, Revenue FYTD.
- **Line chart**: X a `dim_date[year]` → `dim_date[month_name]` hierarchy, values Total Revenue and Revenue LY.
- **Donut**: Total Revenue by `dim_product[category]`.
- **Clustered bar**: Total Revenue by `dim_store[region]`.
- **Map** (optional): `dim_store[city]` sized by Total Revenue.

### Page 2: Store Performance
- **Matrix**: rows `dim_store[store_name]`; values Total Revenue, Revenue YoY %, Profit Margin %, Revenue per Sq Ft, Store Revenue Rank. Add conditional formatting (data bars on revenue, colour scale on YoY %).
- **Scatter**: X Store Sq Ft, Y Total Revenue, legend `store_type`. This shows that Express stores earn the most per square foot.
- **Matrix heatmap**: rows `store_type`, columns `dim_date[day_name]`, values Avg Daily Revenue, background colour scale.

### Page 3: Products & Promotions
- **Table**: product_name, Total Revenue, Cumulative Revenue Share %, ABC Class, sorted by revenue (Pareto view).
- **Treemap**: category → sub_category by Total Revenue.
- **Clustered column**: X `Sales[promo_band]` (Route A) or `Sales[discount_pct]`, Y Avg Units per Day, legend category.
- **Card pair**: Promo Revenue Share %, Discount % of Gross.

### Page 4: Festivals & Seasonality
- **Line chart**: X `dim_date[days_to_diwali]` (filter −30 to 10), Y Avg Daily Revenue, legend `dim_date[year]`. Diwali aligns at 0 even though its date moves every year.
- **Card**: Festive Uplift x.
- **Column chart**: Avg Daily Revenue by `dim_date[holiday_name]` (filter out blank).
- **Line**: Total Revenue and Revenue 28D Moving Avg by date.

### Page 5: Demand Forecast (ML)
- **Slicer**: `ForecastEval[model_name]` (compare the Ensemble against the Seasonal Naive baseline).
- **KPI cards**: Forecast Accuracy %, WAPE %, Forecast Bias %.
- **Line chart**: X `dim_date[full_date]`, values Actual Units (Test) and Forecast Units. The Oct–Dec 2025 test period includes Diwali.
- **Bar**: Forecast Accuracy % by `dim_product[category]`.
- **Line / table**: Next 28D Forecast Units by date and store, from `FutureForecast`.
- **Text box**: "Forecasts are made 28 days ahead; the model reaches ~96% of the maximum achievable accuracy (noise ceiling)."

## 6. Refresh
Route A: **Home → Refresh** after re-running the ETL and training scripts. Route B: re-run the scripts, then refresh (the CSV paths stay the same).

## Interview talking points
- A star schema keeps the fact table narrow (integer surrogate keys), and yearly partitions let 2025-only queries skip older data.
- Materialized views pre-aggregate 584k rows for fast dashboards.
- Forecasts are written back into the warehouse (`fact_forecast`), so the business sees actual vs forecast in the same tool.
