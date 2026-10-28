"""Central configuration. Every script imports paths and constants from here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXPORTS = ROOT / "data" / "exports"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SQL_DIR = ROOT / "sql"

RAW_FILE = DATA_RAW / "retail_sales_raw.csv"
CLEAN_FILE = DATA_PROCESSED / "retail_sales_clean.csv"
FEATURES_FILE = DATA_PROCESSED / "features.pkl"

for d in (DATA_RAW, DATA_PROCESSED, DATA_EXPORTS, MODELS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Data range ------------------------------------------------------------
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"
RANDOM_SEED = 42

# ---- Forecasting setup -----------------------------------------------------
# We forecast 28 days ahead. Every lag/rolling feature is shifted by >= HORIZON
# days, so each prediction only uses information available 28 days earlier.
# That makes the offline test score an honest estimate of real 4-week-ahead
# performance (no look-ahead leakage).
HORIZON = 28
VAL_START = "2025-07-02"   # validation: Jul-Sep 2025 (used for tuning / early stopping)
TEST_START = "2025-10-01"  # test: Oct-Dec 2025 (includes Diwali, the hardest period)

# ---- PostgreSQL ------------------------------------------------------------
DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", 5432)),
    "dbname": os.getenv("PGDATABASE", "retail_dw"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "postgres"),
}


def db_url() -> str:
    c = DB_CONFIG
    return f"postgresql+psycopg2://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['dbname']}"
