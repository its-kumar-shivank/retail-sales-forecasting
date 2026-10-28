"""Generate a realistic daily retail sales dataset for "BharatMart", a fictional
10-store Indian retail chain (2022-2025, 40 products, ~584k rows).

Demand model (multiplicative, per store x product x day):

    lambda = base_product * store_size * trend * weekly * yearly * festival
             * promo_lift * post_promo_dip * month_start * latent_noise

    units ~ NegativeBinomial(mean=lambda)   (Poisson with gamma-distributed rate)

The raw CSV then gets real-world dirt injected (duplicates, missing values, typos,
inconsistent labels, mixed date formats) so that the cleaning step is meaningful.

Run:  python src/generate_data.py            (standard 10-store dataset -> data/raw/retail_sales_raw.csv)
      python src/generate_data.py --large    (17 stores, ~1M rows -> data/raw/retail_sales_large.csv)
"""
import argparse
import json

import numpy as np
import pandas as pd

from calendar_india import FESTIVALS, festival_dates, mega_sale_windows
from config import DATA_RAW, END_DATE, RANDOM_SEED, RAW_FILE, REPORTS_DIR, START_DATE, TEST_START

rng = np.random.default_rng(RANDOM_SEED)

# --------------------------------------------------------------------------
# Master data
# --------------------------------------------------------------------------
STORES = [
    # id, name, city, state, region, type, size_sqft
    ("S01", "BharatMart Connaught Place", "New Delhi", "Delhi", "North", "Hypermarket", 42000),
    ("S02", "BharatMart Hazratganj", "Lucknow", "Uttar Pradesh", "North", "Supermarket", 18000),
    ("S03", "BharatMart MI Road", "Jaipur", "Rajasthan", "North", "Express", 6000),
    ("S04", "BharatMart Andheri", "Mumbai", "Maharashtra", "West", "Hypermarket", 45000),
    ("S05", "BharatMart Koregaon Park", "Pune", "Maharashtra", "West", "Supermarket", 20000),
    ("S06", "BharatMart CG Road", "Ahmedabad", "Gujarat", "West", "Express", 7000),
    ("S07", "BharatMart Indiranagar", "Bengaluru", "Karnataka", "South", "Hypermarket", 40000),
    ("S08", "BharatMart T Nagar", "Chennai", "Tamil Nadu", "South", "Supermarket", 19000),
    ("S09", "BharatMart Banjara Hills", "Hyderabad", "Telangana", "South", "Supermarket", 21000),
    ("S10", "BharatMart Park Street", "Kolkata", "West Bengal", "East", "Supermarket", 17000),
]
# Added only with --large, to reach ~10 lakh rows without changing the standard dataset
EXTRA_STORES = [
    ("S11", "BharatMart Salt Lake", "Kolkata", "West Bengal", "East", "Hypermarket", 38000),
    ("S12", "BharatMart Patna Junction", "Patna", "Bihar", "East", "Express", 6500),
    ("S13", "BharatMart Sector 17", "Chandigarh", "Chandigarh", "North", "Supermarket", 16000),
    ("S14", "BharatMart Adajan", "Surat", "Gujarat", "West", "Supermarket", 18500),
    ("S15", "BharatMart Jubilee Hills", "Hyderabad", "Telangana", "South", "Hypermarket", 41000),
    ("S16", "BharatMart MG Road", "Kochi", "Kerala", "South", "Express", 5500),
    ("S17", "BharatMart Civil Lines", "Nagpur", "Maharashtra", "West", "Supermarket", 17500),
]
STORE_TYPE_MULT = {"Hypermarket": 1.65, "Supermarket": 1.0, "Express": 0.55}

# id, name, category, sub_category, brand, price (INR, Jan 2022), base daily units at an avg supermarket
PRODUCTS = [
    ("P01", "Basmati Rice 5kg", "Grocery & Staples", "Rice & Grains", "FreshFarm", 649, 38),
    ("P02", "Whole Wheat Atta 10kg", "Grocery & Staples", "Flour", "FreshFarm", 449, 45),
    ("P03", "Toor Dal 1kg", "Grocery & Staples", "Pulses", "DesiHarvest", 159, 70),
    ("P04", "Refined Sunflower Oil 1L", "Grocery & Staples", "Edible Oil", "GoldenDrop", 145, 85),
    ("P05", "Iodised Salt 1kg", "Grocery & Staples", "Spices & Salt", "DesiHarvest", 28, 95),
    ("P06", "Sugar 1kg", "Grocery & Staples", "Sugar", "SweetLeaf", 48, 90),
    ("P07", "Garam Masala 100g", "Grocery & Staples", "Spices & Salt", "MasalaKing", 85, 40),
    ("P08", "Masoor Dal 1kg", "Grocery & Staples", "Pulses", "DesiHarvest", 119, 42),
    ("P09", "Cola 2L", "Beverages & Snacks", "Soft Drinks", "FizzUp", 99, 60),
    ("P10", "Mango Drink 1.2L", "Beverages & Snacks", "Juices", "TropiCool", 75, 48),
    ("P11", "Potato Chips 150g", "Beverages & Snacks", "Chips", "CrunchCo", 50, 80),
    ("P12", "Namkeen Mixture 400g", "Beverages & Snacks", "Namkeen", "CrunchCo", 110, 55),
    ("P13", "Assam Tea 500g", "Beverages & Snacks", "Tea & Coffee", "ChaiGhar", 265, 35),
    ("P14", "Instant Coffee 100g", "Beverages & Snacks", "Tea & Coffee", "BrewBean", 330, 18),
    ("P15", "Kaju Katli Box 500g", "Beverages & Snacks", "Sweets", "MithaasHouse", 499, 12),
    ("P16", "Cream Biscuits 300g", "Beverages & Snacks", "Biscuits", "CrunchCo", 40, 90),
    ("P17", "Herbal Shampoo 340ml", "Personal Care", "Hair Care", "PureGlow", 285, 22),
    ("P18", "Neem Soap Pack of 4", "Personal Care", "Bath", "PureGlow", 160, 35),
    ("P19", "Toothpaste 200g", "Personal Care", "Oral Care", "SmileBright", 115, 40),
    ("P20", "Moisturising Lotion 400ml", "Personal Care", "Skin Care", "SilkSkin", 349, 14),
    ("P21", "Face Wash 150ml", "Personal Care", "Skin Care", "SilkSkin", 199, 20),
    ("P22", "Hair Oil 300ml", "Personal Care", "Hair Care", "PureGlow", 175, 26),
    ("P23", "Non-Stick Tawa", "Home & Kitchen", "Cookware", "KitchenPro", 899, 5),
    ("P24", "Pressure Cooker 5L", "Home & Kitchen", "Cookware", "KitchenPro", 2199, 3.5),
    ("P25", "Steel Water Bottle 1L", "Home & Kitchen", "Storage", "UrbanBasics", 449, 9),
    ("P26", "Bedsheet Double", "Home & Kitchen", "Furnishing", "HomeNest", 1299, 4.5),
    ("P27", "LED Diya Lights 10m", "Home & Kitchen", "Decor", "HomeNest", 349, 4),
    ("P28", "Storage Containers Set", "Home & Kitchen", "Storage", "UrbanBasics", 699, 7),
    ("P29", "Bluetooth Earbuds", "Electronics", "Audio", "SonicWave", 1799, 4),
    ("P30", "Power Bank 20000mAh", "Electronics", "Mobile Accessories", "VoltEdge", 1499, 4.5),
    ("P31", "Mixer Grinder 750W", "Electronics", "Kitchen Appliances", "KitchenPro", 3499, 2.2),
    ("P32", "Electric Kettle 1.5L", "Electronics", "Kitchen Appliances", "VoltEdge", 999, 3.5),
    ("P33", "Smart LED Bulb", "Electronics", "Lighting", "VoltEdge", 499, 6),
    ("P34", "Fast Charger 33W", "Electronics", "Mobile Accessories", "VoltEdge", 899, 5),
    ("P35", "Men's Cotton Kurta", "Apparel", "Ethnic Wear", "Rangrez", 999, 6),
    ("P36", "Women's Printed Saree", "Apparel", "Ethnic Wear", "Rangrez", 1499, 5),
    ("P37", "Kids T-Shirt Pack of 3", "Apparel", "Kidswear", "UrbanBasics", 699, 7),
    ("P38", "Men's Denim Jeans", "Apparel", "Western Wear", "UrbanBasics", 1299, 5.5),
    ("P39", "Women's Kurti", "Apparel", "Ethnic Wear", "Rangrez", 799, 7.5),
    ("P40", "Cotton Socks Pack of 5", "Apparel", "Innerwear", "UrbanBasics", 299, 12),
]
COST_RATIO = {"Grocery & Staples": 0.84, "Beverages & Snacks": 0.70, "Personal Care": 0.62,
              "Home & Kitchen": 0.64, "Electronics": 0.78, "Apparel": 0.55}

# Day-of-week multipliers, Mon..Sun
WEEKLY = {
    "Grocery & Staples":  [0.93, 0.90, 0.93, 0.97, 1.03, 1.15, 1.12],
    "Beverages & Snacks": [0.90, 0.88, 0.90, 0.95, 1.08, 1.20, 1.15],
    "Personal Care":      [0.95, 0.93, 0.95, 0.98, 1.02, 1.10, 1.07],
    "Home & Kitchen":     [0.85, 0.83, 0.87, 0.92, 1.02, 1.28, 1.25],
    "Electronics":        [0.82, 0.80, 0.85, 0.90, 1.05, 1.32, 1.30],
    "Apparel":            [0.80, 0.78, 0.83, 0.90, 1.05, 1.35, 1.33],
}
# Month multipliers Jan..Dec (smoothly interpolated to daily)
YEARLY = {
    "Grocery & Staples":  [1.00, 0.98, 1.00, 1.00, 1.02, 1.00, 0.98, 1.00, 1.02, 1.05, 1.07, 1.02],
    "Beverages & Snacks": [0.85, 0.90, 1.05, 1.25, 1.35, 1.20, 1.00, 0.95, 0.95, 1.00, 1.05, 1.02],
    "Personal Care":      [1.03, 0.97, 1.00, 1.03, 1.03, 0.98, 0.96, 0.97, 1.00, 1.04, 1.08, 1.07],
    "Home & Kitchen":     [0.95, 0.92, 0.95, 0.95, 0.95, 0.92, 0.90, 0.95, 1.00, 1.15, 1.20, 1.02],
    "Electronics":        [1.00, 0.90, 0.92, 0.97, 1.02, 0.95, 0.90, 0.95, 1.00, 1.18, 1.22, 1.05],
    "Apparel":            [1.10, 1.00, 0.95, 0.90, 0.95, 0.90, 0.85, 0.90, 0.95, 1.20, 1.28, 1.18],
}
# Festival peak multipliers by category: (days_before, days_after, peak)
FESTIVAL_EFFECT = {
    "diwali": (15, 0, {"Electronics": 2.4, "Apparel": 2.2, "Home & Kitchen": 2.0,
                       "Beverages & Snacks": 1.8, "Grocery & Staples": 1.4, "Personal Care": 1.3}),
    "dussehra": (5, 0, {"Apparel": 1.4, "Electronics": 1.3, "Home & Kitchen": 1.15,
                        "Beverages & Snacks": 1.1, "Grocery & Staples": 1.08, "Personal Care": 1.05}),
    "holi": (4, -1, {"Beverages & Snacks": 1.6, "Personal Care": 1.4, "Grocery & Staples": 1.2,
                     "Apparel": 1.15, "Home & Kitchen": 1.0, "Electronics": 1.0}),
    "eid": (6, -1, {"Apparel": 1.5, "Grocery & Staples": 1.3, "Beverages & Snacks": 1.3,
                    "Personal Care": 1.15, "Home & Kitchen": 1.05, "Electronics": 1.05}),
    "rakhi": (5, 0, {"Beverages & Snacks": 1.6, "Apparel": 1.2, "Personal Care": 1.2,
                     "Electronics": 1.1, "Home & Kitchen": 1.05, "Grocery & Staples": 1.05}),
}
# Special product-level boosts (Diwali lights & sweets explode around Diwali)
PRODUCT_DIWALI_BOOST = {"P27": 4.0, "P15": 2.5}
PROMO_ELASTICITY = {"Grocery & Staples": 1.6, "Beverages & Snacks": 2.2, "Personal Care": 2.4,
                    "Home & Kitchen": 2.8, "Electronics": 3.4, "Apparel": 3.0}


def smooth_yearly(dates: pd.DatetimeIndex, month_mult: list[float]) -> np.ndarray:
    """Interpolate 12 mid-month multipliers into a smooth cyclic daily curve."""
    doy = dates.dayofyear.values
    mids = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    xs = np.concatenate([mids - 365, mids, mids + 365])
    ys = np.tile(month_mult, 3)
    return np.interp(doy, xs, ys)


def festival_multiplier(dates: pd.DatetimeIndex, category: str) -> np.ndarray:
    mult = np.ones(len(dates))
    for fest, (before, _after, peaks) in FESTIVAL_EFFECT.items():
        peak = peaks[category]
        for fd in festival_dates(fest):
            delta = (dates - fd).days.values
            ramp = (delta >= -before) & (delta <= 0)
            # linear ramp: 1.0 at `before` days out -> peak on the day before the festival
            mult[ramp] *= 1 + (peak - 1) * (1 - np.abs(delta[ramp] + 1) / before).clip(0, 1)
            if fest == "diwali":
                mult[(delta >= 1) & (delta <= 3)] *= 0.75   # post-Diwali lull
            if fest == "holi":
                mult[delta == 0] *= 0.55                     # shops half-closed on Holi
    # Christmas / New Year party season
    md = dates.month * 100 + dates.day
    xmas = (md >= 1220) & (md <= 1231)
    xmas_peak = {"Beverages & Snacks": 1.3, "Electronics": 1.15, "Apparel": 1.15}.get(category, 1.05)
    mult[xmas] *= xmas_peak
    # National holidays: +10% footfall
    for m, d in [(1, 26), (8, 15), (10, 2)]:
        mult[(dates.month == m) & (dates.day == d)] *= 1.10
    return mult


def build_promotions(dates: pd.DatetimeIndex, n_stores: int, prod_cats: list[str]):
    """Return discount array (D, S, P) in [0,1] and promo flag.
    Two sources: chain-wide mega sales + random store-level campaigns."""
    D, S, P = len(dates), n_stores, len(prod_cats)
    disc = np.zeros((D, S, P))
    # chain-wide sales
    for start, end, _, cat in mega_sale_windows():
        mask_d = (dates >= start) & (dates <= end)
        if not mask_d.any():
            continue
        for p, c in enumerate(prod_cats):
            if cat is None or c == cat:
                d = rng.choice([0.15, 0.20, 0.25]) if cat is None else rng.choice([0.30, 0.40])
                disc[mask_d, :, p] = np.maximum(disc[mask_d, :, p], d)
    # random store x product campaigns: ~1 start every 45 days, 3-10 days long
    for s in range(S):
        for p in range(P):
            t = int(rng.integers(0, 40))
            while t < D:
                dur = int(rng.integers(3, 11))
                d = rng.choice([0.10, 0.15, 0.20, 0.25, 0.30], p=[0.3, 0.3, 0.2, 0.12, 0.08])
                disc[t:t + dur, s, p] = np.maximum(disc[t:t + dur, s, p], d)
                t += dur + int(rng.geometric(1 / 40))
    return disc


def main(large: bool = False):
    if large:
        STORES.extend(EXTRA_STORES)
    out_file = DATA_RAW / "retail_sales_large.csv" if large else RAW_FILE
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    D, S, P = len(dates), len(STORES), len(PRODUCTS)
    t_years = ((dates - dates[0]).days.values / 365.25)[:, None, None]
    prod_cats = [p[2] for p in PRODUCTS]

    # ---------------- demand components (D, S, P) ----------------
    base = np.array([p[6] for p in PRODUCTS])[None, None, :]
    store_mult = np.array([STORE_TYPE_MULT[s[5]] * rng.uniform(0.9, 1.1) for s in STORES])[None, :, None]
    store_growth = rng.uniform(0.03, 0.12, S)[None, :, None]
    prod_growth = rng.normal(0.02, 0.05, P)[None, None, :]
    trend = np.exp((store_growth + prod_growth) * t_years)

    dow = dates.dayofweek.values
    weekly = np.stack([np.array(WEEKLY[c])[dow] for c in prod_cats], axis=1)[:, None, :]
    # Express stores serve commuters: weekly pattern is half as strong
    is_express = np.array([s[5] == "Express" for s in STORES])[None, :, None]
    weekly = np.where(is_express, 1 + (weekly - 1) * 0.5, weekly)

    yearly = np.stack([smooth_yearly(dates, YEARLY[c]) for c in prod_cats], axis=1)[:, None, :]
    fest = np.stack([festival_multiplier(dates, c) for c in prod_cats], axis=1)
    for pid, boost in PRODUCT_DIWALI_BOOST.items():
        p = int(pid[1:]) - 1
        for fd in festival_dates("diwali"):
            delta = (dates - fd).days.values
            win = (delta >= -12) & (delta <= 0)
            fest[win, p] *= 1 + (boost - 1) * (1 - np.abs(delta[win] + 1) / 12).clip(0, 1)
    fest = fest[:, None, :]

    month_start = np.where(dates.day.values <= 5, 1.06, 1.0)[:, None, None]

    disc = build_promotions(dates, S, prod_cats)
    elasticity = np.array([PROMO_ELASTICITY[c] for c in prod_cats])[None, None, :]
    promo_lift = np.exp(elasticity * disc)
    # post-promo dip: customers stocked up -> 3 days of lower demand after a campaign ends
    ended = (disc[:-1] > 0) & (disc[1:] == 0)
    post = np.zeros_like(disc, dtype=bool)
    for k in range(1, 4):
        post[k:] |= ended[: D - k]
    dip = np.where(post & (disc == 0), 0.9, 1.0)

    # latent noise: slow AR(1) per series (demand drift) + daily store traffic shock
    phi, sigma = 0.985, 0.02
    latent = np.zeros((D, S, P))
    eps = rng.normal(0, sigma, (D, S, P))
    latent[0] = rng.normal(0, sigma / np.sqrt(1 - phi**2), (S, P))
    for d in range(1, D):
        latent[d] = phi * latent[d - 1] + eps[d]
    store_shock = rng.normal(0, 0.05, (D, S, 1))
    noise = np.exp(latent + store_shock)

    lam = base * store_mult * trend * weekly * yearly * fest * month_start * promo_lift * dip * noise

    # Negative binomial draw: gamma-mixed Poisson (shape k controls over-dispersion)
    k = 25.0
    units = rng.poisson(rng.gamma(k, lam / k)).astype(float)

    # stockouts: rare 1-3 day windows where shelves are empty-ish
    so = rng.random((D, S, P)) < 0.0015
    for _ in range(2):
        so[1:] |= so[:-1] & (rng.random((D - 1, S, P)) < 0.5)
    units[so] = np.floor(units[so] * rng.uniform(0, 0.3, so.sum()))

    # ---------------- prices ----------------
    base_price = np.array([p[5] for p in PRODUCTS], dtype=float)
    # price revision every April: +3-7%
    revisions = np.clip(((dates.year.values - 2022) * 12 + dates.month.values - 4) // 12 + 1, 0, None)
    infl = rng.uniform(1.03, 1.07, (5, P))
    price_idx = np.cumprod(np.vstack([np.ones(P), infl]), axis=0)[revisions]   # (D, P)
    price = np.round(base_price[None, :] * price_idx)
    cost = np.round(price * np.array([COST_RATIO[c] for c in prod_cats])[None, :] * rng.uniform(0.97, 1.03, P), 2)

    # ---------------- flatten to long table ----------------
    di, si, pi = np.meshgrid(np.arange(D), np.arange(S), np.arange(P), indexing="ij")
    di, si, pi = di.ravel(), si.ravel(), pi.ravel()
    st = pd.DataFrame(STORES, columns=["store_id", "store_name", "city", "state", "region", "store_type", "store_size_sqft"])
    pr = pd.DataFrame(PRODUCTS, columns=["product_id", "product_name", "category", "sub_category", "brand", "_p", "_b"])

    df = pd.DataFrame({
        "transaction_date": dates.values[di],
        **{c: st[c].values[si] for c in st.columns},
        **{c: pr[c].values[pi] for c in ["product_id", "product_name", "category", "sub_category", "brand"]},
        "unit_price": price[di, pi],
        "unit_cost": cost[di, pi],
        "discount_pct": np.round(disc.ravel() * 100).astype(int),
        "is_promo": (disc.ravel() > 0).astype(int),
        "units_sold": units.ravel(),
    })
    df["revenue"] = np.round(df.units_sold * df.unit_price * (1 - df.discount_pct / 100), 2)

    # Oracle benchmark: error of a "perfect" model that knows the true expected demand.
    # This is the irreducible-noise floor for the test period -> no model can beat it.
    lam_flat = lam.ravel()
    test_mask = df.transaction_date >= pd.Timestamp(TEST_START)
    y, yhat = df.units_sold[test_mask].values, lam_flat[test_mask.values]
    oracle = {"test_period_start": TEST_START,
              "oracle_wape_item_store_day": float(np.abs(y - yhat).sum() / y.sum()),
              "note": "Best achievable WAPE given the random noise in the data (true expected demand known)."}
    if not large:  # don't overwrite the benchmark that belongs to the standard dataset
        (REPORTS_DIR / "oracle_benchmark.json").write_text(json.dumps(oracle, indent=2))

    df = inject_dirt(df)
    df.to_csv(out_file, index=False)
    print(f"Wrote {out_file}  rows={len(df):,}  size={out_file.stat().st_size/1e6:.1f} MB")
    print(f"Oracle WAPE on test period: {oracle['oracle_wape_item_store_day']:.4f}")


def inject_dirt(df: pd.DataFrame) -> pd.DataFrame:
    """Make the data look like a real POS export."""
    n = len(df)
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"]).dt.strftime("%Y-%m-%d")
    # 1) ~1% of dates exported in DD-MM-YYYY by an older POS system
    idx = rng.choice(n, n // 100, replace=False)
    df.loc[idx, "transaction_date"] = pd.to_datetime(df.loc[idx, "transaction_date"]).dt.strftime("%d-%m-%Y")
    # 2) inconsistent category labels
    idx = rng.choice(n, n // 150, replace=False)
    df.loc[idx, "category"] = df.loc[idx, "category"].str.upper()
    idx = rng.choice(n, n // 150, replace=False)
    df.loc[idx, "category"] = "  " + df.loc[idx, "category"].str.lower() + " "
    # 3) city name variants
    df.loc[df.sample(frac=0.3, random_state=1).index.intersection(df.index[df.city == "Bengaluru"]), "city"] = "Bangalore"
    df.loc[df.sample(frac=0.2, random_state=2).index.intersection(df.index[df.city == "Mumbai"]), "city"] = "Bombay"
    # 4) missing values
    df.loc[rng.choice(n, n // 100, replace=False), "unit_price"] = np.nan
    df.loc[rng.choice(n, n // 200, replace=False), "units_sold"] = np.nan
    df.loc[rng.choice(n, n // 300, replace=False), "store_name"] = np.nan
    # 5) data-entry errors: negative quantities and an extra zero typed
    df.loc[rng.choice(n, 250, replace=False), "units_sold"] *= -1
    typo = rng.choice(np.where(df.units_sold.fillna(0) >= 5)[0], 120, replace=False)
    df.loc[typo, "units_sold"] *= 10
    # 6) exact duplicate rows (~0.4%) from a double upload
    dups = df.sample(frac=0.004, random_state=3)
    df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=4).reset_index(drop=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true", help="17 stores, ~1M rows, written to retail_sales_large.csv")
    main(ap.parse_args().large)
