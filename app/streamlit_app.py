"""BharatMart Sales Forecasting app.

Run from the project root:   streamlit run app/streamlit_app.py

Tabs
  1. Forecast        pick a store + product, see history and the next 28 days,
                     try a what-if promotion, download the forecast
  2. Business view   KPIs and trends from the data you loaded
  3. Model report    how accurate the model is and how it was validated
Upload your own CSV in the sidebar (same columns as data/processed/retail_sales_clean.csv).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import CLEAN_FILE, FIGURES_DIR, HORIZON, MODELS_DIR, REPORTS_DIR  # noqa: E402
from features import REQUIRED  # noqa: E402
from preprocess import clean  # noqa: E402
from forecast import forecast_future, load_bundle  # noqa: E402

st.set_page_config(page_title="BharatMart Sales Forecasting", page_icon="📈", layout="wide")


# ------------------------------------------------------------------ cached loaders
@st.cache_resource
def get_model():
    return load_bundle(MODELS_DIR / "production_model.pkl")


@st.cache_data
def load_default():
    return pd.read_csv(CLEAN_FILE, parse_dates=["date"])


MIN_DAYS = 60


@st.cache_data
def load_upload(file) -> pd.DataFrame:
    df = pd.read_csv(file, dtype={"transaction_date": str})
    if "transaction_date" in df.columns and "date" not in df.columns:
        # raw POS export: run the same cleaning as the pipeline (dates, negatives, typos, gaps)
        n_days = df["transaction_date"].nunique()
        if n_days < MIN_DAYS:
            raise ValueError(f"Only {n_days} distinct dates in the file; at least {MIN_DAYS} "
                             "days of history per store and product are needed to forecast.")
        df, _ = clean(df)
        return df
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["date"])
    return df


@st.cache_data(show_spinner="Forecasting the next 28 days for every store and product ...")
def forecast_all(df: pd.DataFrame, _bundle, plan_key: str, promo_plan: pd.DataFrame | None):
    return forecast_future(_bundle, df[REQUIRED], HORIZON, promo_plan)


def inr(x: float) -> str:
    if abs(x) >= 1e7:
        return f"₹{x/1e7:,.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x/1e5:,.2f} L"
    return f"₹{x:,.0f}"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("📈 BharatMart")
st.sidebar.caption("Retail Sales Forecasting & Analytics")
upload = st.sidebar.file_uploader("Upload sales CSV (optional)", type="csv",
                                  help="Needs columns: " + ", ".join(REQUIRED))
if upload is not None:
    try:
        data = load_upload(upload)
    except Exception as e:  # bad file: show the reason instead of a stack trace
        st.sidebar.error(f"Could not use this file: {e}")
        st.stop()
    missing = [c for c in REQUIRED if c not in data.columns]
    if missing:
        st.sidebar.error(f"Missing columns: {', '.join(missing)}")
        st.stop()
    if data.empty:
        st.sidebar.error("No rows with a valid date were found in the file.")
        st.stop()
    st.sidebar.success(f"Loaded {len(data):,} rows")
else:
    data = load_default()

bundle = get_model()
unknown = set(data.store_id.astype(str)) - set(bundle["cat_levels"]["store_id"])
if unknown:
    st.sidebar.warning(f"Stores not seen in training: {', '.join(sorted(unknown))}. "
                       "Their forecasts rely only on history features and will be less accurate.")

span_days = (data.date.max() - data.date.min()).days
if span_days < 420:
    st.sidebar.warning("The model uses last-year features, so it needs ~14 months of history "
                       "for full accuracy.")

st.sidebar.markdown(f"**Model:** {bundle['name']}  \n**Trained until:** {bundle['trained_until']}  \n"
                    f"**Horizon:** {bundle['horizon']} days")

tab_fc, tab_biz, tab_model = st.tabs(["🔮 Forecast", "📊 Business view", "🧪 Model report"])

# ------------------------------------------------------------------ forecast tab
with tab_fc:
    stores = data[["store_id", "store_name"]].drop_duplicates().sort_values("store_id") \
        if "store_name" in data else data[["store_id"]].drop_duplicates().assign(store_name=lambda d: d.store_id)
    products = data[["product_id", "product_name"]].drop_duplicates().sort_values("product_id") \
        if "product_name" in data else data[["product_id"]].drop_duplicates().assign(product_name=lambda d: d.product_id)

    c1, c2 = st.columns(2)
    store = c1.selectbox("Store", stores.store_id, format_func=lambda s: f"{s} · {stores.set_index('store_id').store_name[s]}")
    product = c2.selectbox("Product", products.product_id,
                           format_func=lambda p: f"{p} · {products.set_index('product_id').product_name[p]}")

    with st.expander("🎯 What-if: plan a promotion for this product in this store"):
        future_dates = pd.date_range(data.date.max() + pd.Timedelta(days=1), periods=HORIZON)
        w1, w2 = st.columns(2)
        disc = w1.select_slider("Discount %", options=[0, 10, 15, 20, 25, 30], value=0)
        rng_ = w2.slider("Promo days", min_value=future_dates[0].date(), max_value=future_dates[-1].date(),
                         value=(future_dates[0].date(), future_dates[6].date()))
    plan = None
    if disc > 0:
        pd_dates = pd.date_range(rng_[0], rng_[1])
        plan = pd.DataFrame({"date": pd_dates, "store_id": store, "product_id": product, "discount_pct": disc})
    plan_key = "none" if plan is None else f"{store}-{product}-{disc}-{rng_}"

    base_fc = forecast_all(data, bundle, "none", None)
    fc = forecast_all(data, bundle, plan_key, plan) if plan is not None else base_fc
    sel = fc[(fc.store_id.astype(str) == store) & (fc.product_id.astype(str) == product)]
    sel_base = base_fc[(base_fc.store_id.astype(str) == store) & (base_fc.product_id.astype(str) == product)]
    hist = data[(data.store_id == store) & (data.product_id == product)].sort_values("date").tail(180)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Next 28 days (units)", f"{sel.forecast_units.sum():,.0f}")
    last28 = hist.tail(28).units_sold.sum()
    k2.metric("Last 28 days (units)", f"{last28:,.0f}",
              f"{100*(sel.forecast_units.sum()/max(last28,1)-1):+.1f}% ahead")
    price = hist.unit_price.iloc[-1]
    k3.metric("Expected revenue", inr((sel.forecast_units * price * (1 - sel.discount_pct / 100)).sum()))
    if plan is not None:
        lift = sel.forecast_units.sum() - sel_base.forecast_units.sum()
        k4.metric("Promo lift (units)", f"{lift:+,.0f}",
                  f"{100*lift/max(sel_base.forecast_units.sum(),1):+.1f}%")
    else:
        k4.metric("Peak day", f"{sel.loc[sel.forecast_units.idxmax(), 'date']:%d %b}",
                  f"{sel.forecast_units.max():,.0f} units")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.date, y=hist.units_sold, name="Actual", line=dict(color="#264653")))
    if plan is not None:
        fig.add_trace(go.Scatter(x=sel_base.date, y=sel_base.forecast_units, name="Forecast (no promo)",
                                 line=dict(color="#8d99ae", dash="dot")))
    fig.add_trace(go.Scatter(x=sel.date, y=sel.forecast_units, name="Forecast",
                             line=dict(color="#2a9d8f", width=3)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      yaxis_title="Units per day", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

    out = sel.assign(date=sel.date.dt.date, forecast_units=sel.forecast_units.round(1))
    c1, c2 = st.columns([3, 1])
    c1.dataframe(out[["date", "discount_pct", "forecast_units"]], hide_index=True, width="stretch", height=250)
    c2.download_button("⬇️ This series (CSV)", out.to_csv(index=False), f"forecast_{store}_{product}.csv")
    c2.download_button("⬇️ All stores & products", fc.to_csv(index=False), "forecast_all.csv")

# ------------------------------------------------------------------ business tab
with tab_biz:
    has_rev = "revenue" in data
    if not has_rev:
        data = data.assign(revenue=data.units_sold * data.unit_price * (1 - data.discount_pct / 100))
    last_year = data.date.max().year
    cur = data[data.date.dt.year == last_year]
    prev = data[data.date.dt.year == last_year - 1]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Revenue {last_year}", inr(cur.revenue.sum()),
              f"{100*(cur.revenue.sum()/max(prev.revenue.sum(),1)-1):+.1f}% YoY" if len(prev) else None)
    k2.metric("Units sold", f"{cur.units_sold.sum():,.0f}")
    if "profit" in data:
        k3.metric("Profit margin", f"{100*cur.profit.sum()/cur.revenue.sum():.1f}%")
    k4.metric("Promo share of days", f"{100*(cur.discount_pct > 0).mean():.1f}%")

    m = data.groupby([data.date.dt.to_period("M").dt.to_timestamp(), "category"]).revenue.sum().reset_index()
    st.plotly_chart(px.area(m, x="date", y="revenue", color="category", title="Monthly revenue by category",
                            height=380), width="stretch")
    c1, c2 = st.columns(2)
    by_store = cur.groupby("store_id").revenue.sum().sort_values().reset_index()
    c1.plotly_chart(px.bar(by_store, x="revenue", y="store_id", orientation="h",
                           title=f"Revenue by store ({last_year})", height=380), width="stretch")
    dow = data.assign(dow=data.date.dt.day_name()).groupby(["category", "dow"]).units_sold.mean().reset_index()
    dow["index"] = dow.units_sold / dow.groupby("category").units_sold.transform("mean")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    c2.plotly_chart(px.density_heatmap(dow, x="dow", y="category", z="index", histfunc="avg",
                                       category_orders={"dow": order}, color_continuous_scale="RdYlGn",
                                       title="Day-of-week demand index", height=380), width="stretch")

# ------------------------------------------------------------------ model tab
with tab_model:
    mfile = MODELS_DIR / "metrics.json"
    if not mfile.exists():
        st.info("Run `python src/train.py` to generate the model report.")
    else:
        mt = json.loads(mfile.read_text())
        b = mt["test_metrics_best_model"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy (1-WAPE)", f"{100*b['Accuracy (1-WAPE)']:.1f}%", help="Store x product x day, test Oct-Dec 2025")
        k2.metric("Noise ceiling", f"{100*(1-mt['oracle_WAPE_noise_floor']):.1f}%",
                  help="Accuracy of a perfect model that knows the true expected demand")
        k3.metric("Share of achievable", f"{100*mt['share_of_achievable_accuracy']:.0f}%")
        k4.metric("R²", f"{b['R2']:.3f}")
        st.caption(f"All forecasts are genuinely {mt['horizon_days']} days ahead: every sales-based feature is at "
                   "least 28 days old. Model chosen on validation data; the test set was used once.")

        lv = pd.DataFrame([{"Level": k, "Accuracy %": round(100 * v["Accuracy (1-WAPE)"], 1)}
                           for k, v in mt["accuracy_by_aggregation_level"].items()])
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(lv, x="Level", y="Accuracy %", text="Accuracy %", range_y=[50, 100],
                               title="Accuracy by aggregation level", height=360), width="stretch")
        comp = pd.DataFrame(mt["model_comparison"])[["model", "test_Accuracy (1-WAPE)", "test_MAE", "test_R2", "test_Bias %"]]
        comp.columns = ["Model", "Accuracy", "MAE", "R²", "Bias %"]
        comp["Accuracy"] = (100 * comp["Accuracy"]).round(1).astype(str) + "%"
        c2.markdown("**Model comparison (test set)**")
        c2.dataframe(comp.round(3), hide_index=True, width="stretch")
        for f in ["10_chain_actual_vs_forecast.png", "12_series_examples.png", "11_feature_importance.png"]:
            if (FIGURES_DIR / f).exists():
                st.image(str(FIGURES_DIR / f), width="stretch")
