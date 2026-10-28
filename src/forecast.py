"""Shared helpers used by train.py and the Streamlit app."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from calendar_india import mega_sale_windows
from config import HORIZON, MODELS_DIR
from features import CAT_COLS, build_features

# Count-like features that get a log1p transform for the linear model
LOG_FEATURES_PREFIX = ("lag_", "rmean_", "rstd_", "rmax_", "ewm_", "dow_mean", "base_nopromo",
                       "ly_", "store_rmean", "product_chain")


# ------------------------------------------------------------------ metrics
def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    wape = np.abs(err).sum() / y.sum()
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "WAPE": float(wape),
        "Accuracy (1-WAPE)": float(1 - wape),
        "R2": float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "Bias %": float(100 * (p.sum() / y.sum() - 1)),
    }


def _log1p_clip(a):
    return np.log1p(np.clip(a, 0, None))


# ------------------------------------------------------------------ linear model
class LogLinearRegression:
    """Linear Regression on log1p(units) with one-hot categoricals.

    Why log space: demand effects are multiplicative (weekend x promo x festival), and
    log turns products into sums, which is what a linear model can represent.
    Why smearing: exp(E[log y]) < E[y] (Jensen's inequality), so naively inverting the log
    under-forecasts. Duan's smearing multiplies back by mean(exp(residual)) to remove that bias."""

    def __init__(self):
        self.pipe, self.smear = None, 1.0

    @staticmethod
    def _build(cols):
        onehot = [c for c in cols if c in CAT_COLS] + ["dow", "month"]
        logc = [c for c in cols if c.startswith(LOG_FEATURES_PREFIX)]
        rest = [c for c in cols if c not in onehot and c not in logc]
        ct = ColumnTransformer([
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), onehot),
            ("log", Pipeline([("l", FunctionTransformer(_log1p_clip)), ("s", StandardScaler())]), logc),
            ("num", StandardScaler(), rest)])
        return Pipeline([("prep", ct), ("lr", LinearRegression())])

    def fit(self, X, y):
        self.pipe = self._build(list(X.columns))
        ly = np.log1p(y)
        self.pipe.fit(X, ly)
        resid = ly - self.pipe.predict(X)
        self.smear = float(np.mean(np.exp(resid)))
        return self

    def predict(self, X):
        return np.exp(self.pipe.predict(X)) * self.smear - 1


# ------------------------------------------------------------------ level-scaled tree models
class LevelScaledModel:
    """Tree model trained on target = units / level, prediction = f(x) * level.

    Why: a tree leaf outputs an average of training targets, so trees cannot predict
    values above what they saw. With ~16%/yr growth, each Diwali is the biggest ever
    and raw-unit trees under-forecast it. The ratio 'this day vs baseline' is stable
    across years, so the tree only learns multipliers and the level carries the trend.
    Training uses sample_weight = level so the loss matches absolute unit errors."""

    def __init__(self, model, level_col: str = "level"):
        self.model, self.level_col = model, level_col

    def predict(self, Z, **kw):
        return self.model.predict(Z, **kw) * Z[self.level_col].to_numpy()

    def __getattr__(self, name):          # expose e.g. feature_importance() of the booster
        return getattr(self.model, name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


# ------------------------------------------------------------------ model inputs
def as_model_input(X: pd.DataFrame, features: list[str], cat_levels: dict, kind: str) -> pd.DataFrame:
    """Tree boosters take pandas categoricals; RF / linear get integer codes."""
    Z = X[features].copy()
    for c in CAT_COLS:
        Z[c] = pd.Categorical(Z[c].astype(str), categories=cat_levels[c])
        if kind == "codes":
            Z[c] = Z[c].cat.codes.astype("int16")
    return Z


def predict_bundle(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Weighted blend of the models in the bundle, clipped at 0 (demand can't be negative)."""
    pred = np.zeros(len(X))
    for name, w in bundle["weights"].items():
        m = bundle["models"][name]
        kind = bundle["input_kind"][name]
        Z = as_model_input(X, bundle["features"], bundle["cat_levels"], kind)
        pred += w * np.clip(m.predict(Z), 0, None)
    return pred


def load_bundle(path=MODELS_DIR / "production_model.pkl") -> dict:
    return joblib.load(path)


# ------------------------------------------------------------------ future frame
def make_future_frame(history: pd.DataFrame, days: int = HORIZON, promo_plan: pd.DataFrame | None = None):
    """Append `days` future rows per series. Promotions: chain-wide scheduled sales are known
    in advance (20% default); pass `promo_plan` (date, store_id, product_id, discount_pct) to
    override with the real promo calendar. Price = last known price."""
    last = history.sort_values("date").groupby(["store_id", "product_id"]).tail(1)
    future_dates = pd.date_range(history.date.max() + pd.Timedelta(days=1), periods=days, freq="D")
    fut = (last.drop(columns=["date"]).assign(key=1)
           .merge(pd.DataFrame({"date": future_dates, "key": 1}), on="key").drop(columns="key"))
    fut["units_sold"] = np.nan
    fut["discount_pct"] = 0
    for start, end, _, cat in mega_sale_windows():
        m = (fut.date >= start) & (fut.date <= end)
        if cat is not None:
            m &= fut.category == cat
        fut.loc[m, "discount_pct"] = 20 if cat is None else 30
    if promo_plan is not None and len(promo_plan):
        # override only the (date, store, product) rows the plan mentions
        plan = promo_plan.set_index(["date", "store_id", "product_id"])["discount_pct"]
        key = pd.MultiIndex.from_frame(fut[["date", "store_id", "product_id"]])
        override = plan.reindex(key).to_numpy()
        fut["discount_pct"] = np.where(np.isnan(override), fut["discount_pct"], override)
    return pd.concat([history, fut[history.columns]], ignore_index=True)


def forecast_future(bundle: dict, history: pd.DataFrame, days: int = HORIZON, promo_plan=None) -> pd.DataFrame:
    if days > bundle["horizon"]:
        raise ValueError(f"model was built for a {bundle['horizon']}-day horizon")
    full = make_future_frame(history, days, promo_plan)
    X, _ = build_features(full, horizon=bundle["horizon"])
    Xf = X[X.date > history.date.max()].copy()
    Xf["forecast_units"] = predict_bundle(bundle, Xf)
    return Xf[["date", "store_id", "product_id", "category", "discount_pct", "forecast_units"]]
