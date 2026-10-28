"""Feature engineering for 28-day-ahead demand forecasting.

Leakage rule: any feature built from past *sales* is shifted by >= HORIZON (28) days.
So the forecast for day t uses sales only up to t-28, which is exactly what is known
when a planner forecasts 4 weeks out. Features that are *planned in advance* (calendar,
festivals, promotion calendar, price list) are used for day t itself.

Implementation: the clean data is a complete date x (store, product) grid, so we pivot
units to a wide matrix (days x 400 series) and compute lags / rolling windows column-wise
with vectorised pandas ops. Sorting the long table by (date, store, product) makes
`wide.values.ravel()` line up row-for-row with the long table.
"""
import numpy as np
import pandas as pd

from calendar_india import FESTIVALS, days_to_event, festival_dates, mega_sale_windows
from config import HORIZON

CAT_COLS = ["store_id", "product_id", "category", "store_type", "region"]
REQUIRED = ["date", "store_id", "product_id", "category", "store_type", "region",
            "units_sold", "discount_pct", "unit_price"]


def _complete_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a full date x store x product grid sorted by (date, store, product)."""
    dates = pd.date_range(df.date.min(), df.date.max(), freq="D")
    series = df[["store_id", "product_id"]].drop_duplicates().sort_values(["store_id", "product_id"])
    full = pd.MultiIndex.from_arrays([
        np.repeat(dates, len(series)),
        np.tile(series.store_id.values, len(dates)),
        np.tile(series.product_id.values, len(dates))], names=["date", "store_id", "product_id"])
    out = df.set_index(["date", "store_id", "product_id"]).reindex(full).reset_index()
    static = ["category", "store_type", "region"]
    out[static] = out.groupby(["store_id", "product_id"])[static].transform(lambda s: s.ffill().bfill())
    out["discount_pct"] = out["discount_pct"].fillna(0)
    out["unit_price"] = out.groupby(["store_id", "product_id"])["unit_price"].transform(lambda s: s.ffill().bfill())
    return out


def _diwali_aligned_positions(dates: pd.DatetimeIndex) -> np.ndarray:
    """For each date, row index of the 'equivalent day last year'.
    Near Diwali (+-45 days) we shift by the gap between this and last year's Diwali
    (e.g. 354 or 384 days) so Diwali lines up with Diwali; otherwise shift 364 (same weekday)."""
    diw = festival_dates("diwali")
    shifts = np.full(len(dates), 364)
    for k in range(1, len(diw)):
        near = np.abs((dates - diw[k]).days) <= 45
        shifts[near] = (diw[k] - diw[k - 1]).days
    return np.arange(len(dates)) - shifts


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.DataFrame(index=dates)
    d["dow"] = dates.dayofweek
    d["dom"] = dates.day
    d["month"] = dates.month
    d["weekofyear"] = dates.isocalendar().week.astype(int).values
    d["dayofyear"] = dates.dayofyear
    d["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    d["is_month_start"] = (dates.day <= 5).astype(int)
    d["doy_sin"] = np.sin(2 * np.pi * dates.dayofyear / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * dates.dayofyear / 365.25)
    ds = pd.Series(dates, index=dates)
    for fest in FESTIVALS:
        d[f"days_to_{fest}"] = days_to_event(ds, festival_dates(fest)).values
    d["days_to_christmas"] = days_to_event(ds, festival_dates("christmas"), clip=15).values
    md = dates.month * 100 + dates.day
    d["is_national_holiday"] = np.isin(md, [126, 815, 1002]).astype(int)
    d["mega_sale"] = 0
    for start, end, _, cat in mega_sale_windows():
        if cat is None:
            d.loc[(dates >= start) & (dates <= end), "mega_sale"] = 1
    return d


def build_features(df: pd.DataFrame, horizon: int = HORIZON) -> tuple[pd.DataFrame, list[str]]:
    """df: clean long table (future rows allowed with units_sold = NaN).
    Returns (feature table sorted by date/store/product, list of feature columns)."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = _complete_grid(df[REQUIRED].copy())
    df = df.sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)
    dates = pd.DatetimeIndex(df.date.unique())
    n_d = len(dates)

    W = df.pivot(index="date", columns=["store_id", "product_id"], values="units_sold")
    P = df.pivot(index="date", columns=["store_id", "product_id"], values="discount_pct")
    PR = df.pivot(index="date", columns=["store_id", "product_id"], values="unit_price")
    n_s = W.shape[1]
    assert n_d * n_s == len(df), "grid not complete"

    F = {}   # name -> wide DataFrame (days x series)
    h = horizon
    S = W.shift(h)                                    # latest sales visible at forecast time

    # --- recent level (lags at multiples of 7 keep the weekday aligned)
    for lag in (h, h + 7, h + 14, h + 21, h + 28):
        F[f"lag_{lag}"] = W.shift(lag)
    for w in (7, 14, 28, 56, 112):
        F[f"rmean_{w}"] = S.rolling(w, min_periods=max(2, w // 2)).mean()
    F["rstd_28"] = S.rolling(28, min_periods=14).std()
    F["rmax_28"] = S.rolling(28, min_periods=14).max()
    F["ewm_a10"] = S.ewm(alpha=0.10).mean()
    # same weekday average over the last 4 visible weeks
    F["dow_mean_4w"] = (W.shift(h) + W.shift(h + 7) + W.shift(h + 14) + W.shift(h + 21)) / 4

    # --- promo-free baseline demand (what the series sells without a discount)
    Wnp = W.where(P == 0).shift(h)
    F["base_nopromo_56"] = Wnp.rolling(56, min_periods=10).mean()

    # --- last year: plain and Diwali-aligned
    F["lag_364"] = W.shift(364)
    cm7 = W.rolling(7, center=True, min_periods=3).mean()
    pos = _diwali_aligned_positions(dates)
    valid = pos >= 0
    aligned = np.full((n_d, n_s), np.nan)
    aligned[valid] = cm7.values[pos[valid]]
    F["ly_aligned_cm7"] = pd.DataFrame(aligned, index=W.index, columns=W.columns)
    # year-over-year growth of the visible 28-day level, and last-year value scaled by it
    rm28_ly = W.shift(h + 364).rolling(28, min_periods=14).mean()
    F["yoy_growth"] = (F["rmean_28"] / rm28_ly.replace(0, np.nan)).clip(0.3, 3)
    F["ly_expected"] = F["ly_aligned_cm7"] * F["yoy_growth"]

    # --- hierarchy signals: whole-store traffic and chain-wide product demand
    stores = W.columns.get_level_values(0)
    prods = W.columns.get_level_values(1)
    store_tot = S.T.groupby(level=0).sum(min_count=1).T.rolling(28, min_periods=14).mean()
    prod_tot = S.T.groupby(level=1).sum(min_count=1).T.rolling(28, min_periods=14).mean()
    F["store_rmean_28"] = pd.DataFrame(store_tot[stores].values, index=W.index, columns=W.columns)
    F["product_chain_rmean_28"] = pd.DataFrame(prod_tot[prods].values, index=W.index, columns=W.columns)

    # --- level: promo-free recent baseline. Tree models predict units / level (see train.py),
    #     and these ratio features describe the forecast day relative to that baseline.
    level = F["base_nopromo_56"].fillna(F["rmean_56"]).fillna(F["rmean_28"]).clip(lower=0.5)
    F["level"] = level
    for c in ("rmean_7", "rmean_28", "dow_mean_4w", "ly_aligned_cm7", "ly_expected", "lag_364", "ewm_a10"):
        F[f"{c}_rel"] = F[c] / level

    # --- promotion calendar (planned -> known for the forecast day)
    Pv = P.values
    on = Pv > 0
    idx = np.arange(n_d)[:, None].repeat(n_s, 1).astype(float)
    last_on = pd.DataFrame(np.where(on, idx, np.nan)).ffill().values
    F["days_since_promo"] = pd.DataFrame(np.where(on, 0, np.nan_to_num(idx - last_on, nan=99)).clip(0, 14),
                                         index=W.index, columns=W.columns)
    run = np.zeros_like(Pv)
    for t in range(n_d):
        run[t] = np.where(on[t], (run[t - 1] if t else 0) + 1, 0)
    F["promo_day_idx"] = pd.DataFrame(run, index=W.index, columns=W.columns)
    chain_share = pd.DataFrame(on, index=W.index, columns=W.columns).T.groupby(level=1).mean().T
    F["chain_promo_share"] = pd.DataFrame(chain_share[prods].values, index=W.index, columns=W.columns)

    # --- price list (planned)
    F["price_vs_28d"] = PR / PR.shift(28)
    F["log_price"] = np.log(PR)
    F["effective_price"] = PR * (1 - P / 100)

    # --- assemble long table
    out = df[["date", "store_id", "product_id", "category", "store_type", "region",
              "units_sold", "discount_pct"]].copy()
    out["is_promo"] = (out.discount_pct > 0).astype(int)
    for name, wide in F.items():
        out[name] = wide.values.ravel().astype("float32")
    cal = calendar_features(dates)
    for c in cal.columns:
        out[c] = np.repeat(cal[c].values, n_s)
    for c in CAT_COLS:
        out[c] = out[c].astype("category")

    feature_cols = [c for c in out.columns if c not in ("date", "units_sold")]
    return out, feature_cols
