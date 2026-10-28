# 🎓 BharatMart Retail Sales Forecasting — Learning Guide

> **Who is this for?** If you're new to Machine Learning, Data Science, or SQL — this guide is built for you. We'll walk through every concept used in this project, step by step, from the very basics to the final deployed app.

---

## 📋 How This Guide is Organized

The learning material is divided into **8 phases**. Each phase has its own folder with a `README.md` that explains the concepts in simple language with examples. You should go through them **in order** — each phase builds on the previous one.

```
learning/
├── learning.md                          ← You are here! (Master Guide)
├── 01-python-and-data-basics/README.md  ← Start here if you're new
├── 02-data-cleaning/README.md
├── 03-exploratory-data-analysis/README.md
├── 04-sql-and-data-warehousing/README.md
├── 05-feature-engineering/README.md
├── 06-machine-learning-models/README.md
├── 07-model-evaluation/README.md
└── 08-deployment-and-visualization/README.md
```

---

## 🗺️ The Big Picture

Before diving into phases, here's what this project does end-to-end:

```
Raw Sales Data → Clean It → Explore It → Store in Database → Build Features → Train ML Models → Evaluate → Deploy App
```

Imagine you're a manager at **BharatMart** (a fictional Indian retail chain). You have:
- **10 stores** across India (Delhi, Mumbai, Bangalore, etc.)
- **40 products** (rice, shampoo, earbuds, t-shirts, etc.)
- **4 years of daily sales data** (2022–2025)

Your job: **Predict how many units of each product each store will sell in the next 28 days** so you can order the right amount of stock. Too much → waste. Too little → empty shelves.

That's exactly what this project builds!

---

## 📊 Phase Overview

### Phase 1: Python & Data Basics
> 📁 [`01-python-and-data-basics/README.md`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/learning/01-python-and-data-basics/README.md)

**What you'll learn:**
- Python libraries used in this project (Pandas, NumPy, Matplotlib)
- How data is stored in DataFrames (think: smart spreadsheets)
- Reading CSV files, filtering rows, grouping data
- Basic statistics: mean, median, standard deviation

**Project files this connects to:**
- [`requirements.txt`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/requirements.txt) — all libraries used
- [`src/config.py`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/src/config.py) — project configuration

**⏱️ Estimated time:** 2–3 hours

---

### Phase 2: Data Cleaning & Preprocessing
> 📁 [`02-data-cleaning/README.md`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/learning/02-data-cleaning/README.md)

**What you'll learn:**
- Why real-world data is messy (typos, missing values, duplicates)
- How to detect and fix data quality issues
- What "rolling median" means and why it beats simple averages
- Filling gaps in time-series data

**Project files this connects to:**
- [`src/generate_data.py`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/src/generate_data.py) — creates the raw (dirty) data
- [`src/preprocess.py`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/src/preprocess.py) — cleans the data
- [`reports/data_quality_report.json`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/reports/data_quality_report.json) — cleaning results

**⏱️ Estimated time:** 2–3 hours

---

### Phase 3: Exploratory Data Analysis (EDA)
> 📁 [`03-exploratory-data-analysis/README.md`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/learning/03-exploratory-data-analysis/README.md)

**What you'll learn:**
- How to ask questions of your data using charts and statistics
- Spotting patterns: weekly cycles, seasonal trends, festival spikes
- Which products sell best? Which stores earn most per square foot?
- Understanding correlation and autocorrelation

**Project files this connects to:**
- [`src/eda.py`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/src/eda.py) — generates all analysis charts
- [`reports/eda_summary.md`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/reports/eda_summary.md) — written insights
- [`reports/figures/`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/reports/figures/) — 8 EDA charts

**⏱️ Estimated time:** 2–3 hours

---

### Phase 4: SQL & Data Warehousing
> 📁 [`04-sql-and-data-warehousing/README.md`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/learning/04-sql-and-data-warehousing/README.md)

**What you'll learn:**
- What is a database and why use one instead of CSV files
- Star schema: fact tables vs dimension tables
- Writing SQL queries: SELECT, JOIN, GROUP BY, window functions
- Real business queries: revenue trends, top products, promotion effectiveness

**Project files this connects to:**
- [`sql/01_schema.sql`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/01_schema.sql) — database structure
- [`sql/02_analytics_queries.sql`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/02_analytics_queries.sql) — 15 business queries
- [`sql/03_bi_views.sql`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/sql/03_bi_views.sql) — materialized views
- [`src/load_warehouse.py`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/src/load_warehouse.py) — loading data into database

**⏱️ Estimated time:** 3–4 hours

---

### Phase 5: Feature Engineering
> 📁 [`05-feature-engineering/README.md`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/learning/05-feature-engineering/README.md)

**What you'll learn:**
- What are "features" and why ML models need them
- Time-series features: lags, rolling averages, trends
- The critical concept of **data leakage** and how to avoid it
- Calendar features: festivals, holidays, day of week
- How this project builds 59 features from raw sales data

**Project files this connects to:**
- [`src/features.py`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/src/features.py) — all feature engineering code

**⏱️ Estimated time:** 3–4 hours

---

### Phase 6: Machine Learning Models
> 📁 [`06-machine-learning-models/README.md`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/learning/06-machine-learning-models/README.md)

**What you'll learn:**
- What is a "model" and how does it learn from data
- Baseline models: Naive and Seasonal Naive
- Linear Regression (in log space with Duan's smearing fix)
- Decision Trees → Random Forest → XGBoost → LightGBM
- Tweedie loss: why regular error metrics fail for sales data
- Ensemble blending: combining models for better results

**Project files this connects to:**
- [`src/train.py`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/src/train.py) — model training pipeline
- [`src/forecast.py`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/src/forecast.py) — making predictions
- [`models/`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/models/) — saved model files

**⏱️ Estimated time:** 4–5 hours

---

### Phase 7: Model Evaluation & Metrics
> 📁 [`07-model-evaluation/README.md`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/learning/07-model-evaluation/README.md)

**What you'll learn:**
- How to measure if a model is good or bad
- Key metrics: MAE, RMSE, WAPE, R², Bias
- Train/Validation/Test splits for time-series (no shuffling!)
- Cross-validation with festivals in mind
- Understanding the "noise ceiling" — the theoretical best accuracy
- Reading the model comparison leaderboard

**Project files this connects to:**
- [`models/metrics.json`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/models/metrics.json) — evaluation results
- [`reports/model_comparison.csv`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/reports/model_comparison.csv) — model leaderboard
- [`reports/oracle_benchmark.json`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/reports/oracle_benchmark.json) — noise ceiling

**⏱️ Estimated time:** 2–3 hours

---

### Phase 8: Deployment & Visualization
> 📁 [`08-deployment-and-visualization/README.md`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/learning/08-deployment-and-visualization/README.md)

**What you'll learn:**
- Building interactive web apps with Streamlit
- Caching: making apps fast by remembering results
- "What-If" analysis: simulating promotions before running them
- Power BI dashboards: DAX measures, star-schema relationships
- How everything connects: from raw data to business decisions

**Project files this connects to:**
- [`app/streamlit_app.py`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/app/streamlit_app.py) — the web application
- [`powerbi/`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/powerbi/) — Power BI dashboard guide
- [`run_pipeline.py`](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/run_pipeline.py) — end-to-end automation

**⏱️ Estimated time:** 2–3 hours

---

## 🔄 The Complete Data Flow

Here's how data moves through the entire project, phase by phase:

```
Phase 1 (Basics)
    You learn: Python, Pandas, NumPy
    ↓
Phase 2 (Cleaning)
    generate_data.py → raw CSV (586k rows, dirty)
    preprocess.py    → clean CSV (584k rows, fixed)
    ↓
Phase 3 (EDA)
    eda.py → 8 charts + insights document
    ↓
Phase 4 (SQL)
    load_warehouse.py → PostgreSQL star schema
    analytics queries → business intelligence
    ↓
Phase 5 (Features)
    features.py → 59 engineered features per row
    ↓
Phase 6 (ML Models)
    train.py → 6 models trained & compared
    ↓
Phase 7 (Evaluation)
    metrics.json → model comparison leaderboard
    Best model: 78.36% accuracy (96% of theoretical max!)
    ↓
Phase 8 (Deployment)
    streamlit_app.py → interactive forecasting app
    Power BI → enterprise dashboards
```

---

## 📚 Prerequisites

Before starting, make sure you have:

| Prerequisite | Level Needed | Where to Learn |
|---|---|---|
| **Python** | Basic (variables, loops, functions) | [Python Tutorial](https://docs.python.org/3/tutorial/) |
| **Terminal/Command Line** | Basic (running commands) | Built into your OS |
| **Git** | Optional (cloning repos) | [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics) |
| **Math** | High school (percentages, averages) | We explain everything else! |

---

## 💡 Tips for Learning

1. **Don't skip phases** — each one builds on the previous
2. **Read the project files** — the links take you directly to the code
3. **Try running the code** — `python run_pipeline.py --skip-db` runs everything
4. **It's okay to not understand everything** — re-read sections as needed
5. **Focus on intuition first** — understand *why* before *how*

---

## 🏁 Let's Begin!

Start with **[Phase 1: Python & Data Basics →](file:///Users/kumarshivank/Downloads/retail-sales-forecasting/learning/01-python-and-data-basics/README.md)**

Happy learning! 🚀
