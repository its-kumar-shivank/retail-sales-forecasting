"""Run the whole project end to end.

    python run_pipeline.py              # all steps (needs PostgreSQL running)
    python run_pipeline.py --skip-db    # everything except the warehouse (ML still works)
"""
import argparse
import subprocess
import sys
import time

STEPS = [
    ("Generate dataset", ["src/generate_data.py"]),
    ("Clean & preprocess", ["src/preprocess.py"]),
    ("Exploratory data analysis", ["src/eda.py"]),
    ("Load PostgreSQL warehouse", ["src/load_warehouse.py", "--export"]),
    ("Train & evaluate models", ["src/train.py"]),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-db", action="store_true")
    ap.add_argument("--skip-generate", action="store_true", help="keep the existing raw CSV")
    args = ap.parse_args()
    for name, cmd in STEPS:
        if args.skip_db and "load_warehouse" in cmd[0]:
            continue
        if args.skip_generate and "generate_data" in cmd[0]:
            continue
        print(f"\n=== {name} ===", flush=True)
        t = time.time()
        subprocess.run([sys.executable, *cmd], check=True)
        print(f"--- {name} done in {time.time() - t:.0f}s")
    print("\nAll done. Launch the app with:  streamlit run app/streamlit_app.py")
