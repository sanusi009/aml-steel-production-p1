"""Usage: python src/scripts/run_pipeline.py"""
import argparse
from pathlib import Path

from pipeline import run

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=ROOT / "data/raw/normalized_train_data.csv")
    p.add_argument("--test", default=ROOT / "data/raw/normalized_test_data.csv")
    p.add_argument("--out", default=ROOT / "results")
    a = p.parse_args()
    print(run(a.train, a.test, a.out).round(4).T.to_string(header=False))
