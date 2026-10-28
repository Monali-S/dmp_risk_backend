"""CLI: validate the correlation model against Monte Carlo truth.

    python run_experiment.py --scenarios 30 --tasks 15 --sims 10000
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from app.evaluation import run_benchmark

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=30)
    ap.add_argument("--tasks", type=int, default=15)
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    res = run_benchmark(a.scenarios, a.tasks, a.sims, a.seed)
    print(json.dumps(res["summary"], indent=2))
    Path("outputs").mkdir(exist_ok=True)
    pd.DataFrame(res["rows"]).to_csv("outputs/benchmark.csv", index=False)
    print("Per-scenario results saved to outputs/benchmark.csv")
