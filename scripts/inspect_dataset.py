"""
Quick dataset inspection script.

Run this FIRST to understand the structure, types, missingness, and target
encoding of networkTraffic.csv before running the full EDA script. The output
of this script informs decisions in scripts/run_eda.py (e.g. how attack_cat
is encoded, which columns are truly categorical, where missing/placeholder
values like '?' live, etc.)

Usage:
    python scripts/inspect_dataset.py
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(REPO_ROOT, "data", "networkTraffic.csv")
FEATURES_PATH = os.path.join(REPO_ROOT, "data", "features.csv")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def main():
    df = pd.read_csv(DATA_PATH)
    features = pd.read_csv(FEATURES_PATH)

    print("=" * 80)
    print("SHAPE")
    print("=" * 80)
    print(df.shape)

    print("\n" + "=" * 80)
    print("DTYPES")
    print("=" * 80)
    print(df.dtypes)

    print("\n" + "=" * 80)
    print("HEAD")
    print("=" * 80)
    print(df.head())

    print("\n" + "=" * 80)
    print("FEATURES.CSV CONTENTS")
    print("=" * 80)
    print(features)

    print("\n" + "=" * 80)
    print("COLUMNS IN networkTraffic.csv NOT DESCRIBED IN features.csv")
    print("=" * 80)
    described = set(features["Name"].str.strip().str.lower())
    actual = set(c.strip().lower() for c in df.columns)
    print("In data but not in features.csv:", sorted(actual - described))
    print("In features.csv but not in data:", sorted(described - actual))

    print("\n" + "=" * 80)
    print("TARGET COLUMN: attack_cat")
    print("=" * 80)
    print("dtype:", df["attack_cat"].dtype)
    print("unique values:", sorted(df["attack_cat"].unique()))
    print("value counts:")
    print(df["attack_cat"].value_counts())

    print("\n" + "=" * 80)
    print("NULL / NaN COUNTS PER COLUMN")
    print("=" * 80)
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "No NaNs found.")

    print("\n" + "=" * 80)
    print("PLACEHOLDER '?' COUNTS PER COLUMN (object columns only)")
    print("=" * 80)
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        n_q = (df[c] == "?").sum()
        if n_q > 0:
            print(f"{c}: {n_q} ({n_q / len(df):.2%})")

    print("\n" + "=" * 80)
    print("CATEGORICAL COLUMN CARDINALITY")
    print("=" * 80)
    for c in obj_cols:
        print(f"{c}: {df[c].nunique()} unique -> {sorted(df[c].unique())[:20]}")

    print("\n" + "=" * 80)
    print("DUPLICATE ROWS (excluding id)")
    print("=" * 80)
    cols_no_id = [c for c in df.columns if c != "id"]
    print("Duplicate rows:", df.duplicated(subset=cols_no_id).sum())

    print("\n" + "=" * 80)
    print("NUMERIC SUMMARY STATS")
    print("=" * 80)
    print(df.describe().T)

    print("\n" + "=" * 80)
    print("CONSTANT / NEAR-CONSTANT COLUMNS (<=1 unique value)")
    print("=" * 80)
    for c in df.columns:
        if df[c].nunique() <= 1:
            print(c)


if __name__ == "__main__":
    main()