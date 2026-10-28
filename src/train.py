"""Step 6: train and evaluate forecasting models.

Protocol (time-based, no shuffling — shuffling would leak the future into training):
  train      2023-03-01 .. 2025-07-01   (first 14 months only feed the lag features)
  validation 2025-07-02 .. 2025-09-30   model selection, early stopping, ensemble weights
  test       2025-10-01 .. 2025-12-31   touched ONCE at the end (includes Diwali)

For the test score every model is refit on train+validation, mimicking production
where you retrain on all available history before forecasting.

Models: naive baselines, Linear Regression, Random Forest, XGBoost, LightGBM, ensemble.
Metric: WAPE = sum|forecast - actual| / sum(actual); Accuracy = 1 - WAPE.
WAPE is preferred over MAPE in retail because MAPE explodes on low-volume days.

Run:  python src/train.py
"""
import gc
import json
import time

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from config import (CLEAN_FILE, DATA_EXPORTS, END_DATE, FIGURES_DIR, HORIZON, MODELS_DIR,
                    RANDOM_SEED, REPORTS_DIR, TEST_START, VAL_START)
from features import CAT_COLS, build_features
from forecast import (LevelScaledModel, LogLinearRegression, as_model_input, forecast_future, metrics,
                      predict_bundle)

TRAIN_START = "2023-03-01"   # Linear / RF need complete last-year features
GBM_START = "2022-03-01"     # boosters handle missing values natively -> 1 extra Diwali to learn from
# LightGBM is tuned with time-series CV. Fold 1 is a festive quarter, fold 2 a normal one:
# validating on Jul-Sep alone (no festival) makes early stopping quit before the model
# has learned the 3-4x Diwali multipliers.
CV_FOLDS = [("2024-10-01", "2024-12-31"), ("2025-07-02", "2025-09-30")]
CV_ROUNDS = 600
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


CKPT = MODELS_DIR / "_checkpoints"
CKPT.mkdir(exist_ok=True)


def stage(name, fn):
    """Run fn() once and cache its result, so an interrupted run resumes where it stopped."""
    path = CKPT / f"{name.replace(' ', '_')}.pkl"
    if path.exists():
        log(f"{name}: loaded from checkpoint")
        return joblib.load(path)
    out = fn()
    joblib.dump(out, path)
    return out


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- model builders
LGB_BASE = dict(objective="tweedie", learning_rate=0.1, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=1.0, cat_smooth=10, verbose=-1, seed=RANDOM_SEED)
LGB_GRID = [
    dict(num_leaves=63,  min_child_samples=50,  tweedie_variance_power=1.2),
    dict(num_leaves=255, min_child_samples=50,  tweedie_variance_power=1.2),
    dict(num_leaves=127, min_child_samples=100, tweedie_variance_power=1.4),
]
XGB_PARAMS = dict(objective="reg:tweedie", tweedie_variance_power=1.2, learning_rate=0.1, max_depth=8,
                  min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
                  random_state=RANDOM_SEED, n_jobs=-1)
RF_PARAMS = dict(n_estimators=60, max_depth=16, min_samples_leaf=10, max_features=0.33, max_samples=0.3,
                 n_jobs=-1, random_state=RANDOM_SEED)


def ratio(X, y):
    """Level-normalised target and weights (see LevelScaledModel)."""
    lv = X["level"].to_numpy()
    return y / lv, lv


def fit_lgb(Xtr, ytr, params, rounds, Xva=None, yva=None):
    r, w = ratio(Xtr, ytr)
    dtr = lgb.Dataset(Xtr, r, weight=w, categorical_feature=CAT_COLS, free_raw_data=True)
    if Xva is None:
        return LevelScaledModel(lgb.train(params, dtr, num_boost_round=rounds))
    rv, wv = ratio(Xva, yva)
    dva = lgb.Dataset(Xva, rv, weight=wv, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return LevelScaledModel(m)


def fit_xgb(Xtr, ytr, rounds, Xva=None, yva=None):
    r, w = ratio(Xtr, ytr)
    if Xva is None:
        return LevelScaledModel(xgb.XGBRegressor(n_estimators=rounds, **XGB_PARAMS).fit(Xtr, r, sample_weight=w))
    rv, wv = ratio(Xva, yva)
    m = xgb.XGBRegressor(n_estimators=rounds, early_stopping_rounds=100, eval_metric="mae", **XGB_PARAMS)
    m.fit(Xtr, r, sample_weight=w, eval_set=[(Xva, rv)], sample_weight_eval_set=[wv], verbose=False)
    return LevelScaledModel(m)


def fit_rf(Xtr, ytr):
    r, w = ratio(Xtr, ytr)
    return LevelScaledModel(RandomForestRegressor(**RF_PARAMS).fit(Xtr, r, sample_weight=w))


def main():
    t0 = time.time()
    df = pd.read_csv(CLEAN_FILE, parse_dates=["date"])
    log("Building features ...")
    X_all, feats = build_features(df)
    cat_levels = {c: [str(v) for v in X_all[c].cat.categories] for c in CAT_COLS}
    X = X_all[X_all.date >= GBM_START].reset_index(drop=True)
    del X_all
    gc.collect()
    tr = (X.date < VAL_START).values                     # boosters
    tr_lin = (tr & (X.date >= TRAIN_START)).values       # linear / random forest
    va = ((X.date >= VAL_START) & (X.date < TEST_START)).values
    te = (X.date >= TEST_START).values
    trva, trva_lin = tr | va, tr_lin | va
    all_lin = (X.date >= TRAIN_START).values
    y = X.units_sold.values
    log(f"rows: train {tr.sum():,} (linear/RF {tr_lin.sum():,}) | val {va.sum():,} | test {te.sum():,} "
        f"| features {len(feats)}")

    Xc = as_model_input(X, feats, cat_levels, "category")  # for LightGBM / XGBoost
    Xn = as_model_input(X, feats, cat_levels, "codes")     # for RF / Linear

    val_pred, test_pred, fitted, input_kind = {}, {}, {}, {}

    # ---- baselines (no training)
    val_pred["Naive (last 4 weeks avg)"] = X.rmean_28[va].values
    test_pred["Naive (last 4 weeks avg)"] = X.rmean_28[te].values
    sn = X.ly_expected.fillna(X.rmean_28)
    val_pred["Seasonal Naive (last year x growth)"] = sn[va].values
    test_pred["Seasonal Naive (last year x growth)"] = sn[te].values

    # ---- Linear Regression
    def run_lr():
        log("Linear Regression ...")
        m = LogLinearRegression().fit(Xn[tr_lin], y[tr_lin])
        pv = np.clip(m.predict(Xn[va]), 0, None)
        m = LogLinearRegression().fit(Xn[trva_lin], y[trva_lin])
        return pv, np.clip(m.predict(Xn[te]), 0, None), m
    val_pred["Linear Regression"], test_pred["Linear Regression"], fitted["Linear Regression"] = stage("Linear Regression", run_lr)
    input_kind["Linear Regression"] = "codes"

    # ---- Random Forest
    def run_rf():
        log("Random Forest ...")
        m = fit_rf(Xn[tr_lin], y[tr_lin])
        pv = m.predict(Xn[va])
        m = fit_rf(Xn[trva_lin], y[trva_lin])
        return pv, m.predict(Xn[te]), m
    val_pred["Random Forest"], test_pred["Random Forest"], fitted["Random Forest"] = stage("Random Forest", run_rf)
    input_kind["Random Forest"] = "codes"

    # ---- XGBoost (early stopping on validation, then refit with the found #rounds)
    def run_xgb():
        log("XGBoost ...")
        m = fit_xgb(Xc[tr], y[tr], 2000, Xc[va], y[va])
        rounds = m.model.best_iteration + 1
        log(f"  XGBoost best rounds: {rounds}")
        pv = m.predict(Xc[va], iteration_range=(0, rounds))
        m = fit_xgb(Xc[trva], y[trva], int(rounds * 1.1))
        return pv, m.predict(Xc[te]), m, rounds
    val_pred["XGBoost"], test_pred["XGBoost"], fitted["XGBoost"], xgb_rounds = stage("XGBoost", run_xgb)
    input_kind["XGBoost"] = "category"

    # ---- LightGBM: hyper-parameters + rounds chosen by time-series CV (festive + normal fold)
    tuning = []
    best = None
    for i, g in enumerate(LGB_GRID):
        params = {**LGB_BASE, **g}
        curves = []
        for k, (fs, fe) in enumerate(CV_FOLDS):
            def run_fold():
                log(f"LightGBM config {g} fold {fs}..{fe} ...")
                trm = (X.date < fs).values
                vm = ((X.date >= fs) & (X.date <= fe)).values
                r, w = ratio(Xc[trm], y[trm])
                rv, wv = ratio(Xc[vm], y[vm])
                ev = {}
                m = lgb.train({**params, "metric": "l1"},
                              lgb.Dataset(Xc[trm], r, weight=w, categorical_feature=CAT_COLS),
                              CV_ROUNDS, valid_sets=[lgb.Dataset(Xc[vm], rv, weight=wv)],
                              callbacks=[lgb.record_evaluation(ev)])
                # weighted L1 on the ratio = sum|y - yhat| / sum(level)  ->  rescale to exact WAPE
                curve = np.array(ev["valid_0"]["l1"]) * wv.sum() / y[vm].sum()
                return curve, LevelScaledModel(m)
            curves.append(stage(f"LightGBM_cfg{i}_fold{k}", run_fold))
        mean_curve = np.mean([c for c, _ in curves], axis=0)
        rounds = int(mean_curve.argmin()) + 1
        fold_w = [float(c[rounds - 1]) for c, _ in curves]
        tuning.append({**g, "best_rounds": rounds, "cv_WAPE_festive_fold": round(fold_w[0], 5),
                       "cv_WAPE_normal_fold": round(fold_w[1], 5), "cv_WAPE_mean": round(float(mean_curve.min()), 5)})
        log(f"  {g} -> rounds {rounds}, CV WAPE festive {fold_w[0]:.4f} / normal {fold_w[1]:.4f}")
        if best is None or mean_curve.min() < best[0]:
            pv = curves[1][1].predict(Xc[va], num_iteration=rounds)   # fold 2 == validation window
            best = (mean_curve.min(), params, rounds, pv)
    _, lgb_params, lgb_rounds, val_pred["LightGBM"] = best

    def run_lgb_refit():
        log("LightGBM refit on train+val ...")
        m = fit_lgb(Xc[trva], y[trva], lgb_params, int(lgb_rounds * 1.1))
        return m.predict(Xc[te]), m
    test_pred["LightGBM"], fitted["LightGBM"] = stage("LightGBM_refit", run_lgb_refit)
    input_kind["LightGBM"] = "category"
    lgb_model = fitted["LightGBM"]

    # ---- Ensemble: weights chosen on validation only
    log("Ensemble weights ...")
    cands = ["LightGBM", "XGBoost", "Random Forest"]
    best_w, best_wape = None, 1e9
    for a in np.arange(0, 1.01, 0.1):
        for b in np.arange(0, 1.01 - a, 0.1):
            w = {"LightGBM": a, "XGBoost": b, "Random Forest": 1 - a - b}
            p = sum(w[k] * val_pred[k] for k in cands)
            s = metrics(y[va], p)["WAPE"]
            if s < best_wape:
                best_w, best_wape = w, s
    best_w = {k: round(float(v), 2) for k, v in best_w.items() if v > 1e-9}
    val_pred["Ensemble"] = sum(w * val_pred[k] for k, w in best_w.items())
    test_pred["Ensemble"] = sum(w * test_pred[k] for k, w in best_w.items())
    log(f"  weights: {best_w}")

    # ---- comparison
    oracle = json.loads((REPORTS_DIR / "oracle_benchmark.json").read_text())["oracle_wape_item_store_day"]
    rows = []
    for name in val_pred:
        mv, mt = metrics(y[va], val_pred[name]), metrics(y[te], test_pred[name])
        rows.append({"model": name, "val_WAPE": mv["WAPE"], **{f"test_{k}": v for k, v in mt.items()}})
    comp = pd.DataFrame(rows).sort_values("val_WAPE").reset_index(drop=True)
    comp.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    best_name = comp.model[0]                                  # chosen on VALIDATION
    log(f"Best model on validation: {best_name}")
    print(comp.round(4).to_string(index=False))

    # ---- deeper evaluation of the chosen model on the test set
    T = X.loc[te, ["date", "store_id", "product_id", "category", "store_type", "units_sold"]].copy()
    T["forecast"] = test_pred[best_name]
    T["seasonal_naive"] = test_pred["Seasonal Naive (last year x growth)"]
    levels = {
        "Store x Product x Day": T,
        "Store x Product x Week": T.groupby(["store_id", "product_id", pd.Grouper(key="date", freq="W")])[["units_sold", "forecast"]].sum(),
        "Store x Day": T.groupby(["store_id", "date"])[["units_sold", "forecast"]].sum(),
        "Category x Day": T.groupby(["category", "date"])[["units_sold", "forecast"]].sum(),
        "Chain x Day": T.groupby("date")[["units_sold", "forecast"]].sum(),
    }
    level_acc = {k: metrics(v.units_sold, v.forecast) for k, v in levels.items()}
    by_cat = {c: metrics(g.units_sold, g.forecast) for c, g in T.groupby("category", observed=True)}
    diw = T[(T.date >= "2025-10-05") & (T.date <= "2025-10-23")]
    diwali = {"model": metrics(diw.units_sold, diw.forecast),
              "seasonal_naive": metrics(diw.units_sold, diw.seasonal_naive)}
    best_test = metrics(T.units_sold, T.forecast)

    summary = {
        "horizon_days": HORIZON,
        "split": {"train": [GBM_START + " (boosters) / " + TRAIN_START + " (linear, RF)", VAL_START], "validation": [VAL_START, TEST_START],
                  "test": [TEST_START, END_DATE]},
        "best_model": best_name,
        "ensemble_weights": best_w,
        "lightgbm_best_params": {k: v for k, v in lgb_params.items() if k in LGB_GRID[0]},
        "lightgbm_tuning": tuning,
        "test_metrics_best_model": best_test,
        "oracle_WAPE_noise_floor": oracle,
        "share_of_achievable_accuracy": (1 - best_test["WAPE"]) / (1 - oracle),
        "accuracy_by_aggregation_level": level_acc,
        "accuracy_by_category": by_cat,
        "diwali_window_2025": diwali,
        "model_comparison": comp.to_dict(orient="records"),
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

    # ---- figures
    make_figures(comp, oracle, T, lgb_model, feats, level_acc, best_name)

    # ---- evaluation bundle (trained on train+val)
    names = list(best_w) if best_name == "Ensemble" else [best_name]
    weights = best_w if best_name == "Ensemble" else {best_name: 1.0}
    joblib.dump({"models": {n: fitted[n] for n in names}, "weights": weights, "input_kind": input_kind,
                 "features": feats, "cat_levels": cat_levels, "horizon": HORIZON,
                 "trained_until": str(pd.Timestamp(TEST_START) - pd.Timedelta(days=1))[:10],
                 "name": best_name}, MODELS_DIR / "evaluation_model.pkl", compress=3)

    # ---- production model: refit the chosen setup on ALL data, forecast the next 28 days
    log("Refitting production model on all data ...")
    builders = {
        "LightGBM": lambda: fit_lgb(Xc, y, lgb_params, int(lgb_rounds * 1.15)),
        "XGBoost": lambda: fit_xgb(Xc, y, int(xgb_rounds * 1.15)),
        "Random Forest": lambda: fit_rf(Xn[all_lin], y[all_lin]),
        "Linear Regression": lambda: LogLinearRegression().fit(Xn[all_lin], y[all_lin]),
    }
    prod_models = {n: stage(f"production_{n}", builders[n]) for n in names}
    prod = {"models": prod_models, "weights": weights, "input_kind": input_kind, "features": feats,
            "cat_levels": cat_levels, "horizon": HORIZON, "trained_until": END_DATE, "name": best_name}
    joblib.dump(prod, MODELS_DIR / "production_model.pkl", compress=3)

    future = forecast_future(prod, df[["date", "store_id", "product_id", "category", "store_type", "region",
                                       "units_sold", "discount_pct", "unit_price"]])
    future.to_csv(DATA_EXPORTS / "future_forecast_28d.csv", index=False)
    T.to_csv(DATA_EXPORTS / "test_predictions.csv", index=False)
    log(f"Future forecast: {future.date.min().date()} .. {future.date.max().date()}, "
        f"{future.forecast_units.sum():,.0f} units")

    write_back_to_warehouse(T, future, best_name)
    log(f"Finished in {(time.time() - t0) / 60:.1f} min")


def make_figures(comp, oracle, T, lgb_model, feats, level_acc, best_name):
    # model comparison
    fig, ax = plt.subplots(figsize=(10, 4.8))
    c = comp.sort_values("test_Accuracy (1-WAPE)")
    colors = ["#2a9d8f" if m == best_name else "#8d99ae" for m in c.model]
    ax.barh(c.model, 100 * c["test_Accuracy (1-WAPE)"], color=colors)
    ax.axvline(100 * (1 - oracle), color="crimson", ls="--", lw=1.5, label=f"Noise ceiling {100*(1-oracle):.1f}%")
    for i, v in enumerate(100 * c["test_Accuracy (1-WAPE)"]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
    ax.set(xlim=(50, 90), xlabel="Test accuracy (1 - WAPE), store x product x day",
           title="Model comparison on Oct-Dec 2025 (28-day-ahead forecasts)")
    ax.legend(loc="lower right")
    fig.savefig(FIGURES_DIR / "09_model_comparison.png"); plt.close(fig)

    # chain-level actual vs forecast
    ch = T.groupby("date")[["units_sold", "forecast", "seasonal_naive"]].sum()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(ch.index, ch.units_sold, label="Actual", color="black", lw=1.8)
    ax.plot(ch.index, ch.forecast, label=f"{best_name} forecast", color="#2a9d8f", lw=1.8)
    ax.plot(ch.index, ch.seasonal_naive, label="Seasonal naive", color="#e76f51", lw=1, ls="--")
    ax.axvline(pd.Timestamp("2025-10-20"), color="grey", ls=":")
    ax.text(pd.Timestamp("2025-10-21"), ch.units_sold.max() * 0.97, "Diwali", color="grey")
    ax.set(title=f"Chain daily units: actual vs forecast (test period) | accuracy "
                 f"{100*level_acc['Chain x Day']['Accuracy (1-WAPE)']:.1f}%", ylabel="Units")
    ax.legend()
    fig.savefig(FIGURES_DIR / "10_chain_actual_vs_forecast.png"); plt.close(fig)

    # feature importance (gain)
    imp = pd.Series(lgb_model.feature_importance("gain"), index=feats).sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(imp.index, imp / imp.sum() * 100, color="#264653")
    ax.set(title="LightGBM feature importance (share of total gain, %)", xlabel="%")
    fig.savefig(FIGURES_DIR / "11_feature_importance.png"); plt.close(fig)

    # example series
    ex = [("S04", "P01", "Fast mover: Basmati Rice @ Mumbai Hyper"), ("S07", "P29", "Slow mover: Earbuds @ Bengaluru Hyper"),
          ("S02", "P36", "Festive: Saree @ Lucknow Super"), ("S01", "P27", "Spiky: Diwali Lights @ Delhi Hyper")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    for ax, (s, p, title) in zip(axes.ravel(), ex):
        g = T[(T.store_id == s) & (T.product_id == p)]
        a = metrics(g.units_sold, g.forecast)["Accuracy (1-WAPE)"]
        ax.plot(g.date, g.units_sold, color="black", lw=1.2, label="Actual")
        ax.plot(g.date, g.forecast, color="#2a9d8f", lw=1.6, label="Forecast")
        ax.set_title(f"{title}  |  acc {100*a:.0f}%", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    axes[0, 0].legend()
    fig.suptitle("Series-level forecasts, test period", y=1.01)
    fig.savefig(FIGURES_DIR / "12_series_examples.png"); plt.close(fig)

    # accuracy by level
    lv = pd.Series({k: 100 * v["Accuracy (1-WAPE)"] for k, v in level_acc.items()})
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(lv.index, lv.values, color="#2a9d8f")
    for i, v in enumerate(lv.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    ax.set(ylim=(50, 100), ylabel="Accuracy (1 - WAPE) %",
           title="Forecast accuracy rises with aggregation (noise averages out)")
    ax.tick_params(axis="x", rotation=15)
    fig.savefig(FIGURES_DIR / "13_accuracy_by_level.png"); plt.close(fig)


def write_back_to_warehouse(T, future, best_name):
    """Store forecasts in fact_forecast so Power BI can report actual vs forecast."""
    try:
        import psycopg2  # noqa: F401
        from load_warehouse import connect, copy_df, export_bi_tables
        conn = connect()
    except Exception as e:  # warehouse is optional for training
        log(f"Warehouse not reachable, skipping write-back ({e.__class__.__name__})")
        return
    sk = pd.read_sql("SELECT store_id, store_key FROM dim_store", conn).set_index("store_id").store_key
    pk = pd.read_sql("SELECT product_id, product_key FROM dim_product", conn).set_index("product_id").product_key

    def frame(d, col, model, actual):
        return pd.DataFrame({
            "date_key": d.date.dt.strftime("%Y%m%d").astype(int),
            "store_key": d.store_id.astype(str).map(sk), "product_key": d.product_id.astype(str).map(pk),
            "model_name": model, "forecast_units": d[col].round(2),
            "actual_units": d[actual].astype("Int64") if actual else pd.array([pd.NA] * len(d), dtype="Int64")})

    out = pd.concat([frame(T, "forecast", best_name, "units_sold"),
                     frame(T, "seasonal_naive", "Seasonal Naive", "units_sold"),
                     frame(future, "forecast_units", best_name, None)])
    with conn.cursor() as cur:
        cur.execute("TRUNCATE fact_forecast")
    conn.commit()
    copy_df(conn, out, "fact_forecast")
    log(f"Wrote {len(out):,} forecast rows to fact_forecast")
    export_bi_tables(conn)
    conn.close()


if __name__ == "__main__":
    main()
