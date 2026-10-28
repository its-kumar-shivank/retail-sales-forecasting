"""Step 2: clean the raw POS export.

Fixes, in order:
  1. Mixed date formats (YYYY-MM-DD and DD-MM-YYYY) -> parsed explicitly, never guessed
  2. Exact duplicate rows and duplicate (date, store, product) keys
  3. Inconsistent labels (category casing/whitespace, Bangalore/Bengaluru, Bombay/Mumbai)
  4. Missing store names -> restored from the store master (store_id is the source of truth)
  5. Missing unit prices -> same product's price on the same day at other stores
  6. Negative quantities and "extra zero" typos -> detected against a local rolling median
  7. Missing quantities -> linear interpolation inside each store-product series
  8. Revenue / cost / profit recomputed from clean fields
Every fix is counted and written to reports/data_quality_report.json.

Run:  python src/preprocess.py
"""
import json

import numpy as np
import pandas as pd

from config import CLEAN_FILE, RAW_FILE, REPORTS_DIR

CATEGORIES = ["Grocery & Staples", "Beverages & Snacks", "Personal Care",
              "Home & Kitchen", "Electronics", "Apparel"]
CITY_FIX = {"Bangalore": "Bengaluru", "Bombay": "Mumbai"}


def parse_dates(s: pd.Series) -> tuple[pd.Series, int]:
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
    return iso.fillna(dmy), int((iso.isna() & dmy.notna()).sum())


def fix_quantity_errors(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Detect data-entry errors per series using a centered 15-day rolling median.

    Why a *local* median and not a global z-score: festival spikes (Diwali = 2-3x normal)
    are real demand and must survive cleaning. A typo with an extra zero is ~10x the
    local level, so a ratio threshold of 6x on the local median separates the two."""
    neg = df["units_sold"] < 0
    report["negative_units_set_to_nan"] = int(neg.sum())
    df.loc[neg, "units_sold"] = np.nan

    g = df.groupby(["store_id", "product_id"])["units_sold"]
    local_med = g.transform(lambda x: x.rolling(15, center=True, min_periods=5).median())
    ratio = df["units_sold"] / local_med.clip(lower=1)
    suspect = (ratio > 6) & (df["units_sold"] >= 30)
    # If dividing by 10 lands back near the local level, it was an extra zero -> correct it
    fixable = suspect & ((df["units_sold"] / 10) / local_med.clip(lower=1)).between(0.3, 3)
    report["extra_zero_typos_corrected"] = int(fixable.sum())
    report["other_outliers_set_to_nan"] = int((suspect & ~fixable).sum())
    df.loc[fixable, "units_sold"] = (df.loc[fixable, "units_sold"] / 10).round()
    df.loc[suspect & ~fixable, "units_sold"] = np.nan
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean a raw POS export (needs a transaction_date column). Returns (clean_df, report)."""
    report = {}
    df = df.copy()
    df["transaction_date"] = df["transaction_date"].astype(str)
    report["raw_rows"] = len(df)

    # 1. dates
    df["date"], report["dates_in_dd_mm_yyyy_format"] = parse_dates(df.pop("transaction_date"))
    report["unparseable_dates_dropped"] = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])

    # 2. duplicates
    n = len(df)
    df = df.drop_duplicates()
    report["exact_duplicates_removed"] = n - len(df)
    n = len(df)
    df = df.drop_duplicates(subset=["date", "store_id", "product_id"], keep="first")
    report["duplicate_keys_removed"] = n - len(df)

    # 3. labels
    canon = {c.lower(): c for c in CATEGORIES}
    cat_clean = df["category"].str.strip().str.lower().map(canon)
    report["category_labels_standardised"] = int((cat_clean != df["category"]).sum())
    df["category"] = cat_clean
    report["city_names_standardised"] = int(df["city"].isin(CITY_FIX).sum())
    df["city"] = df["city"].replace(CITY_FIX)

    # 4. store names from master (mode per store_id)
    report["missing_store_names_filled"] = int(df["store_name"].isna().sum())
    master = df.dropna(subset=["store_name"]).groupby("store_id")["store_name"].agg(lambda s: s.mode()[0])
    df["store_name"] = df["store_id"].map(master)

    # 5. unit prices: same product, same day, other stores (prices are chain-wide)
    report["missing_prices_filled"] = int(df["unit_price"].isna().sum())
    df["unit_price"] = df["unit_price"].fillna(df.groupby(["product_id", "date"])["unit_price"].transform("median"))
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    df["unit_price"] = df.groupby(["store_id", "product_id"])["unit_price"].ffill().bfill()

    # 6. quantity errors
    report["missing_units_in_raw"] = int(df["units_sold"].isna().sum())
    df = fix_quantity_errors(df, report)

    # 7. complete the date x store x product grid, then interpolate gaps within each series
    stores, products = df["store_id"].unique(), df["product_id"].unique()
    full = pd.MultiIndex.from_product(
        [sorted(stores), sorted(products), pd.date_range(df["date"].min(), df["date"].max())],
        names=["store_id", "product_id", "date"])
    df = df.set_index(["store_id", "product_id", "date"])
    report["missing_grid_rows_added"] = int(len(full) - len(df))
    df = df.reindex(full).reset_index()
    attr_cols = ["store_name", "city", "state", "region", "store_type", "store_size_sqft"]
    df[attr_cols] = df.groupby("store_id")[attr_cols].transform(lambda s: s.ffill().bfill())
    prod_cols = ["product_name", "category", "sub_category", "brand"]
    df[prod_cols] = df.groupby("product_id")[prod_cols].transform(lambda s: s.ffill().bfill())
    for c in ["unit_price", "unit_cost"]:
        df[c] = df.groupby(["store_id", "product_id"])[c].transform(lambda s: s.ffill().bfill())
    df[["discount_pct", "is_promo"]] = df[["discount_pct", "is_promo"]].fillna(0)

    report["units_imputed_by_interpolation"] = int(df["units_sold"].isna().sum())
    df["units_sold"] = (df.groupby(["store_id", "product_id"])["units_sold"]
                        .transform(lambda s: s.interpolate(limit_direction="both")).round())

    # 8. recompute money columns from clean fields
    df["discount_pct"] = df["discount_pct"].astype(int)
    df["is_promo"] = df["is_promo"].astype(int)
    df["units_sold"] = df["units_sold"].astype(int)
    df["gross_sales"] = (df["units_sold"] * df["unit_price"]).round(2)
    df["discount_amount"] = (df["gross_sales"] * df["discount_pct"] / 100).round(2)
    df["revenue"] = (df["gross_sales"] - df["discount_amount"]).round(2)
    df["cost"] = (df["units_sold"] * df["unit_cost"]).round(2)
    df["profit"] = (df["revenue"] - df["cost"]).round(2)

    cols = ["date", "store_id", "store_name", "city", "state", "region", "store_type", "store_size_sqft",
            "product_id", "product_name", "category", "sub_category", "brand",
            "unit_price", "unit_cost", "discount_pct", "is_promo", "units_sold",
            "gross_sales", "discount_amount", "revenue", "cost", "profit"]
    df = df[cols].sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)

    assert df.isna().sum().sum() == 0, "cleaning left NaNs behind"
    assert not df.duplicated(["date", "store_id", "product_id"]).any()
    assert (df["units_sold"] >= 0).all()

    report["clean_rows"] = len(df)
    report["date_range"] = [str(df["date"].min().date()), str(df["date"].max().date())]
    return df, report


def main():
    df, report = clean(pd.read_csv(RAW_FILE, dtype={"transaction_date": str}))
    df.to_csv(CLEAN_FILE, index=False)
    (REPORTS_DIR / "data_quality_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {CLEAN_FILE}")


if __name__ == "__main__":
    main()
