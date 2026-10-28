"""Step 4: ETL clean CSV -> PostgreSQL star schema.

  Extract   : read data/processed/retail_sales_clean.csv
  Transform : build dimensions, assign surrogate keys, map facts to keys
  Load      : bulk COPY into partitioned fact_sales, build BI views, validate totals

Fact rows are loaded with COPY FROM STDIN (a single streamed bulk operation) instead of
row-by-row INSERTs: each INSERT is a network round trip + parse + plan, COPY is one.

Run:  python src/load_warehouse.py            (DB settings in config.py / PG* env vars)
      python src/load_warehouse.py --export   (also dump BI tables to data/exports/*.csv)
"""
import argparse
import io
import time

import numpy as np
import pandas as pd
import psycopg2

from calendar_india import days_to_event, festival_dates, holiday_table
from config import CLEAN_FILE, DATA_EXPORTS, DB_CONFIG, SQL_DIR


def connect():
    return psycopg2.connect(**DB_CONFIG)


def ensure_database():
    """Create the target database if it doesn't exist yet."""
    cfg = dict(DB_CONFIG, dbname="postgres")
    conn = psycopg2.connect(**cfg)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_CONFIG["dbname"],))
        if cur.fetchone() is None:
            cur.execute(f'CREATE DATABASE "{DB_CONFIG["dbname"]}"')
            print(f"Created database {DB_CONFIG['dbname']}")
    conn.close()


def run_sql_file(conn, path):
    with conn.cursor() as cur:
        cur.execute(path.read_text())
    conn.commit()


def copy_df(conn, df: pd.DataFrame, table: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(f"COPY {table} ({','.join(df.columns)}) FROM STDIN WITH (FORMAT csv)", buf)
    conn.commit()


def build_dim_date(start: str, end: str) -> pd.DataFrame:
    d = pd.DataFrame({"full_date": pd.date_range(start, end, freq="D")})
    dt = d.full_date.dt
    hol = holiday_table().set_index("date")["holiday_name"]
    fy_start = np.where(dt.month >= 4, dt.year, dt.year - 1)
    d["date_key"] = dt.strftime("%Y%m%d").astype(int)
    d["day_of_month"] = dt.day
    d["day_name"] = dt.day_name()
    d["day_of_week"] = dt.dayofweek + 1
    d["is_weekend"] = dt.dayofweek >= 5
    d["week_of_year"] = dt.isocalendar().week.astype(int).values
    d["month"] = dt.month
    d["month_name"] = dt.month_name()
    d["quarter"] = dt.quarter
    d["year"] = dt.year
    d["fiscal_year"] = [f"FY{y}-{str(y + 1)[-2:]}" for y in fy_start]
    d["fiscal_quarter"] = "Q" + (((dt.month - 4) % 12) // 3 + 1).astype(str)
    d["holiday_name"] = d.full_date.map(hol)
    d["is_holiday"] = d.holiday_name.notna()
    d["days_to_diwali"] = days_to_event(d.full_date, festival_dates("diwali"))
    d["is_festive_season"] = d.days_to_diwali.between(-15, 0)
    return d


def promo_band(p: int) -> str:
    if p == 0:
        return "No Promo"
    if p <= 15:
        return "Light (10-15%)"
    if p <= 25:
        return "Medium (20-25%)"
    return "Deep (30%+)"


def export_bi_tables(conn):
    tables = ["dim_date", "dim_store", "dim_product", "dim_promotion",
              "mv_monthly_sales", "mv_daily_store_sales", "vw_forecast_vs_actual"]
    for t in tables:
        df = pd.read_sql(f"SELECT * FROM {t}", conn)
        df.to_csv(DATA_EXPORTS / f"{t}.csv", index=False)
        print(f"  exported {t:<24} {len(df):>8,} rows")


def main(export: bool):
    t0 = time.time()
    ensure_database()
    conn = connect()

    print("Creating schema ...")
    run_sql_file(conn, SQL_DIR / "01_schema.sql")

    df = pd.read_csv(CLEAN_FILE, parse_dates=["date"])

    # ---- dimensions
    dim_date = build_dim_date(df.date.min(), "2026-12-31")  # extra year so future forecasts have keys
    copy_df(conn, dim_date[["date_key", "full_date", "day_of_month", "day_name", "day_of_week", "is_weekend",
                            "week_of_year", "month", "month_name", "quarter", "year", "fiscal_year",
                            "fiscal_quarter", "is_holiday", "holiday_name", "days_to_diwali",
                            "is_festive_season"]], "dim_date")

    stores = (df[["store_id", "store_name", "city", "state", "region", "store_type", "store_size_sqft"]]
              .drop_duplicates("store_id").sort_values("store_id"))
    copy_df(conn, stores, "dim_store")
    products = (df[["product_id", "product_name", "category", "sub_category", "brand"]]
                .drop_duplicates("product_id").sort_values("product_id"))
    copy_df(conn, products, "dim_product")

    promos = pd.DataFrame({"discount_pct": sorted(set(range(0, 55, 5)) | set(df.discount_pct.unique()))})
    promos["promo_key"] = range(1, len(promos) + 1)
    promos["is_promo"] = promos.discount_pct > 0
    promos["promo_band"] = promos.discount_pct.map(promo_band)
    copy_df(conn, promos[["promo_key", "discount_pct", "is_promo", "promo_band"]], "dim_promotion")

    # ---- surrogate key lookup (read back what Postgres assigned)
    store_key = pd.read_sql("SELECT store_id, store_key FROM dim_store", conn).set_index("store_id").store_key
    product_key = pd.read_sql("SELECT product_id, product_key FROM dim_product", conn).set_index("product_id").product_key
    promo_key = promos.set_index("discount_pct").promo_key

    fact = pd.DataFrame({
        "date_key": df.date.dt.strftime("%Y%m%d").astype(int),
        "store_key": df.store_id.map(store_key),
        "product_key": df.product_id.map(product_key),
        "promo_key": df.discount_pct.map(promo_key),
        **{c: df[c] for c in ["units_sold", "unit_price", "unit_cost", "gross_sales",
                              "discount_amount", "revenue", "cost", "profit"]},
    })
    assert fact.isna().sum().sum() == 0, "unmapped surrogate keys"

    print(f"Loading fact_sales ({len(fact):,} rows) via COPY ...")
    copy_df(conn, fact, "fact_sales")

    print("Building BI views ...")
    run_sql_file(conn, SQL_DIR / "03_bi_views.sql")
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("ANALYZE")
    conn.autocommit = False

    # ---- validation: warehouse must reconcile exactly with the source file
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), SUM(units_sold), ROUND(SUM(revenue), 2) FROM fact_sales")
        n, units, rev = cur.fetchone()
    assert n == len(df), f"row count mismatch {n} vs {len(df)}"
    assert units == df.units_sold.sum(), "units mismatch"
    assert abs(float(rev) - df.revenue.sum()) < 1.0, "revenue mismatch"
    print(f"Validated: {n:,} rows, {units:,} units, ₹{float(rev):,.2f} revenue — matches source.")

    if export:
        print("Exporting BI tables to data/exports ...")
        export_bi_tables(conn)
    conn.close()
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="export BI tables to CSV for Power BI")
    main(ap.parse_args().export)
