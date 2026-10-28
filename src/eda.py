"""Step 3: Exploratory Data Analysis. Saves figures to reports/figures and key numbers
to reports/eda_summary.md. The findings here directly drive feature engineering:
weekly pattern -> day-of-week features, Diwali ramp -> days_to_diwali, promo lift ->
discount features, year-over-year growth -> trend-aware rolling features.

Run:  python src/eda.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from calendar_india import festival_dates
from config import CLEAN_FILE, FIGURES_DIR, REPORTS_DIR

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})
CR = 1e7  # 1 crore


def save(fig, name):
    fig.savefig(FIGURES_DIR / name)
    plt.close(fig)


def main():
    df = pd.read_csv(CLEAN_FILE, parse_dates=["date"])
    notes = ["# EDA Summary\n"]

    # 1. Monthly revenue trend by year
    m = df.groupby([df.date.dt.year.rename("year"), df.date.dt.month.rename("month")]).revenue.sum().div(CR).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.lineplot(data=m, x="month", y="revenue", hue="year", marker="o", ax=ax, palette="viridis")
    ax.set(title="Monthly revenue by year (₹ crore)", xlabel="Month", ylabel="₹ crore", xticks=range(1, 13))
    save(fig, "01_monthly_revenue_by_year.png")
    yearly = df.groupby(df.date.dt.year).revenue.sum().div(CR)
    growth = yearly.pct_change().mul(100).round(1)
    notes.append("## Revenue and growth\n")
    for y in yearly.index:
        g = "" if pd.isna(growth[y]) else f" (YoY {growth[y]:+.1f}%)"
        notes.append(f"- {y}: ₹{yearly[y]:.1f} crore{g}")

    # 2. Category mix and margin
    cat = df.groupby("category").agg(revenue=("revenue", "sum"), profit=("profit", "sum"))
    cat["margin_pct"] = cat.profit / cat.revenue * 100
    cat = cat.sort_values("revenue", ascending=False)
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.bar(cat.index, cat.revenue / CR, color=sns.color_palette("deep")[0])
    ax1.set_ylabel("Revenue (₹ crore)")
    ax2 = ax1.twinx()
    ax2.plot(cat.index, cat.margin_pct, color="crimson", marker="o", lw=2)
    ax2.set_ylabel("Profit margin %", color="crimson")
    ax2.grid(False)
    ax1.set_title("Revenue and profit margin by category")
    ax1.tick_params(axis="x", rotation=20)
    save(fig, "02_category_revenue_margin.png")
    notes.append("\n## Category mix\n")
    for c, r in cat.iterrows():
        notes.append(f"- {c}: ₹{r.revenue/CR:.1f} crore, margin {r.margin_pct:.1f}%")

    # 3. Day-of-week index by category (1.0 = category average)
    dow = df.groupby(["category", df.date.dt.dayofweek.rename("dow")]).units_sold.mean().unstack()
    dow = dow.div(dow.mean(axis=1), axis=0)
    dow.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(dow, annot=True, fmt=".2f", cmap="RdYlGn", center=1, ax=ax)
    ax.set(title="Day-of-week demand index (1.00 = weekly average)", xlabel="", ylabel="")
    save(fig, "03_day_of_week_heatmap.png")
    notes.append("\n## Weekly pattern\n")
    notes.append(f"- Weekend uplift is strongest in Apparel (Sat index {dow.loc['Apparel','Sat']:.2f}) and "
                 f"weakest in Personal Care (Sat index {dow.loc['Personal Care','Sat']:.2f}).")

    # 4. Diwali effect aligned on days-to-Diwali (the festival date moves every year)
    rows = []
    daily_cat = df.groupby(["date", "category"]).units_sold.sum().reset_index()
    for d in festival_dates("diwali"):
        w = daily_cat[(daily_cat.date >= d - pd.Timedelta(days=30)) & (daily_cat.date <= d + pd.Timedelta(days=10))].copy()
        if w.empty:
            continue
        w["days_to_diwali"] = (w.date - d).dt.days
        base = w[w.days_to_diwali <= -25].groupby("category").units_sold.mean()
        w["index"] = w.units_sold / w.category.map(base)
        rows.append(w)
    dw = pd.concat(rows)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.lineplot(data=dw, x="days_to_diwali", y="index", hue="category", ax=ax, errorbar=None)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set(title="Demand around Diwali (avg of 4 years, 1.0 = normal)", xlabel="Days relative to Diwali",
           ylabel="Demand index")
    save(fig, "04_diwali_effect.png")
    peak = dw[dw.days_to_diwali.between(-3, -1)].groupby("category")["index"].mean().sort_values(ascending=False)
    notes.append("\n## Festival effect (Diwali)\n")
    notes.append("- Peak demand index in the 3 days before Diwali: " +
                 ", ".join(f"{c} {v:.1f}x" for c, v in peak.items()))
    notes.append("- Diwali moves between Oct 20 and Nov 12, so a same-day-last-year lag misses it; "
                 "a `days_to_diwali` feature is needed.")

    # 5. Promotion lift: compare promo vs non-promo days within the same store-product-month
    key = [df.store_id, df.product_id, df.date.dt.to_period("M")]
    base = df[df.is_promo == 0].groupby([df.store_id, df.product_id, df.date.dt.to_period("M")]).units_sold.mean()
    tmp = df[df.is_promo == 1].copy()
    tmp["base"] = base.reindex(pd.MultiIndex.from_arrays(
        [tmp.store_id, tmp.product_id, tmp.date.dt.to_period("M")])).values
    tmp = tmp[tmp.base > 0]
    tmp["lift"] = tmp.units_sold / tmp.base
    lift = tmp.groupby(["category", "discount_pct"]).lift.median().reset_index()
    lift = lift[lift.discount_pct.isin([10, 15, 20, 25, 30])]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.lineplot(data=lift, x="discount_pct", y="lift", hue="category", marker="o", ax=ax)
    ax.set(title="Promotion lift vs discount depth (median, same store-product-month)",
           xlabel="Discount %", ylabel="Units lift (x)")
    save(fig, "05_promo_lift.png")
    l20 = lift[lift.discount_pct == 20].set_index("category").lift.sort_values(ascending=False)
    notes.append("\n## Promotions\n")
    notes.append(f"- {df.is_promo.mean()*100:.1f}% of store-product-days are on promotion.")
    notes.append("- Median lift at 20% off: " + ", ".join(f"{c} {v:.2f}x" for c, v in l20.items()))

    # 6. Store performance
    st = df.groupby(["store_id", "store_name", "store_type", "store_size_sqft"]).revenue.sum().reset_index()
    st["rev_per_sqft"] = st.revenue / st.store_size_sqft / 4  # per year (4 years of data)
    st = st.sort_values("revenue", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.barplot(data=st, y="store_name", x=st.revenue / CR, hue="store_type", dodge=False, ax=axes[0])
    axes[0].set(title="Total revenue by store (₹ crore)", xlabel="₹ crore", ylabel="")
    sns.barplot(data=st.sort_values("rev_per_sqft", ascending=False), y="store_name", x="rev_per_sqft",
                hue="store_type", dodge=False, ax=axes[1])
    axes[1].set(title="Revenue per sq ft per year (₹)", xlabel="₹ / sq ft / year", ylabel="")
    save(fig, "06_store_performance.png")
    best = st.sort_values("rev_per_sqft", ascending=False).iloc[0]
    notes.append("\n## Stores\n")
    notes.append(f"- Top store by revenue: {st.iloc[0].store_name} (₹{st.iloc[0].revenue/CR:.1f} crore).")
    notes.append(f"- Best space productivity: {best.store_name} ({best.store_type}), "
                 f"₹{best.rev_per_sqft:,.0f} per sq ft per year. Small Express stores punch above their size.")

    # 7. Autocorrelation of chain daily units -> which lags carry signal
    chain = df.groupby("date").units_sold.sum()
    lags = np.arange(1, 400)
    acf = [chain.autocorr(l) for l in lags]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(lags, acf, width=1.0)
    for l in (7, 28, 364):
        ax.axvline(l, color="crimson", ls="--", lw=1)
        ax.text(l + 3, max(acf) * 0.95, f"lag {l}", color="crimson")
    ax.set(title="Autocorrelation of chain-level daily units", xlabel="Lag (days)", ylabel="ACF")
    save(fig, "07_autocorrelation.png")
    notes.append("\n## Autocorrelation\n")
    notes.append(f"- ACF at lag 7 = {chain.autocorr(7):.2f}, lag 28 = {chain.autocorr(28):.2f}, "
                 f"lag 364 = {chain.autocorr(364):.2f}. Strong weekly and yearly memory -> lags at "
                 "multiples of 7 and lag 364 are the core forecasting features.")

    # 8. Distribution of daily units per series (why we use a Tweedie/Poisson loss)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(data=df, x="category", y="units_sold", ax=ax, showfliers=False)
    ax.set(title="Daily units per store-product (outliers hidden)", xlabel="", ylabel="Units / day")
    ax.tick_params(axis="x", rotation=20)
    save(fig, "08_units_distribution.png")
    zero_share = (df.units_sold == 0).mean() * 100
    notes.append("\n## Target distribution\n")
    notes.append(f"- Units are non-negative counts, right-skewed, {zero_share:.1f}% zeros. Fast movers "
                 "(grocery ~100/day) and slow movers (electronics ~7/day) differ by >10x, so the model "
                 "uses a Tweedie loss (built for skewed non-negative targets) instead of plain MSE.")

    (REPORTS_DIR / "eda_summary.md").write_text("\n".join(notes) + "\n")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
