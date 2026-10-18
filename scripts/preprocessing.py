"""
Preprocessing pipeline for the UNSW-NB15-style network traffic dataset
(Assignment 4, ensemble learning option).

This script is informed directly by scripts/inspect_dataset.py and
scripts/run_eda.py findings. Each transformation below is justified by a
specific EDA observation - see comments and the printed summary.

Outputs (all under <repo_root>/data/processed/):
  - X_train.csv, X_test.csv, y_train.csv, y_test.csv
  - preprocessing_report.txt (human-readable summary of what was done)
  - rare_proto_categories.csv (which proto values were grouped into 'other')

Usage:
    python scripts/preprocessing.py
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(REPO_ROOT, "data", "networkTraffic.csv")
OUT_DIR = os.path.join(REPO_ROOT, "data", "processed")

TARGET = "attack_cat"
ID_COL = "id"

CATEGORICAL_COLS = ["proto", "service", "state"]

# Minimum relative frequency for a 'proto' category to be kept as its own
# one-hot column. Categories below this threshold are grouped into 'other'.
# Justification: proto has 133 unique values (EDA section 5); many occur
# only a handful of times (e.g. 'ggp', 'ip' at 132 rows = 0.05%). One-hot
# encoding all 133 would create a very sparse, high-dimensional feature
# space dominated by near-constant columns.
RARE_PROTO_THRESHOLD = 0.001  # 0.1%

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Optional, heavily-caveated attack_cat -> name mapping, inferred by matching
# this dataset's class counts against the published UNSW-NB15 distribution.
# Used ONLY for the human-readable preprocessing report, NOT for modeling.
ATTACK_CAT_NAME_MAP = {
    0: "Normal",
    9: "Generic",
    4: "Exploits",
    6: "Fuzzers",
    3: "DoS",
    1: "Reconnaissance",
    5: "Analysis",
    2: "Backdoor",
    8: "Shellcode",
    7: "Worms",
}


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    return pd.read_csv(DATA_PATH)


def drop_irrelevant_columns(df, log):
    # id is a pure row index (confirmed in EDA section 1) - carries no
    # predictive information and must be dropped before modeling.
    df = df.drop(columns=[ID_COL])
    log.append("Dropped 'id' column (row index, no predictive value).")
    return df


def fix_is_ftp_login(df, log):
    # EDA section 7: is_ftp_login is documented as Binary but contains
    # {0, 1, 2, 4} in the raw data. We binarize via >0 -> 1, treating any
    # nonzero value as "ftp login occurred", consistent with the documented
    # semantics ("if the ftp session is accessed by user and password then 1
    # else 0"). This is a judgement call - document it explicitly in your
    # report as a data quality correction.
    before_unique = sorted(df["is_ftp_login"].unique().tolist())
    df["is_ftp_login"] = (df["is_ftp_login"] > 0).astype(int)
    log.append(
        f"Binarized 'is_ftp_login': raw unique values were {before_unique}, "
        f"recoded to {{0,1}} via (value > 0)."
    )
    return df


def group_rare_proto(df, log):
    # EDA section 5: proto has 133 unique values, many extremely rare.
    # We group any category with relative frequency below
    # RARE_PROTO_THRESHOLD into 'other' BEFORE one-hot encoding, to keep the
    # resulting feature space manageable and avoid near-constant one-hot
    # columns that provide little signal and inflate dimensionality.
    freqs = df["proto"].value_counts(normalize=True)
    rare_categories = freqs[freqs < RARE_PROTO_THRESHOLD].index.tolist()
    kept_categories = freqs[freqs >= RARE_PROTO_THRESHOLD].index.tolist()

    df["proto"] = df["proto"].where(~df["proto"].isin(rare_categories), "other")

    log.append(
        f"Grouped {len(rare_categories)} rare 'proto' categories "
        f"(relative frequency < {RARE_PROTO_THRESHOLD:.1%}) into 'other'. "
        f"Kept {len(kept_categories)} categories as-is "
        f"(including the new 'other' bucket, final unique count = "
        f"{df['proto'].nunique()})."
    )

    rare_df = freqs[freqs < RARE_PROTO_THRESHOLD].reset_index()
    rare_df.columns = ["proto_category", "relative_frequency"]
    rare_df.to_csv(os.path.join(OUT_DIR, "rare_proto_categories.csv"), index=False)

    return df


def handle_service_placeholder(df, log):
    # EDA section 2: service is '?' for 54.85% of rows, and this missingness
    # rate varies strongly by attack_cat class (1.5% to 100%), indicating
    # missingness itself is informative. We therefore keep '?' as an
    # explicit category (renamed 'unknown' for clarity) rather than
    # imputing or dropping rows/columns.
    df["service"] = df["service"].replace("?", "unknown")
    log.append(
        "Replaced service=='?' with explicit category 'unknown' "
        "(kept as a category rather than imputed, since missingness rate "
        "varies strongly by attack_cat class - see EDA section 2)."
    )
    return df


def handle_state_anomaly(df, log):
    # EDA section 5: state contains a single row with value 'no', which is
    # not part of the documented state list. With only 1 row affected, we
    # keep it as its own category rather than drop the row (dropping a
    # single row has negligible effect on class balance either way, but
    # keeping it avoids silently discarding data without strong cause).
    n_anomalous = (df["state"] == "no").sum()
    log.append(
        f"'state' contains {n_anomalous} row(s) with undocumented value "
        f"'no'. Kept as its own category (negligible row count, no strong "
        f"justification to drop)."
    )
    return df


def encode_categoricals(df, log):
    # One-hot encoding chosen for proto/service/state because:
    #   - these are nominal (no inherent ordering) - label/ordinal encoding
    #     would impose a false ordinal relationship that tree splits could
    #     exploit spuriously.
    #   - tree ensembles handle resulting sparse binary features natively
    #     without needing scaling.
    #   - cardinality is now manageable after rare-category grouping
    #     (proto) and placeholder handling (service).
    before_cols = df.shape[1]
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS)
    after_cols = df.shape[1]
    log.append(
        f"One-hot encoded {CATEGORICAL_COLS}. Column count went from "
        f"{before_cols} to {after_cols} (+{after_cols - before_cols} "
        f"one-hot columns)."
    )
    return df


def split_data(df, log):
    # EDA section 4: attack_cat has severe class imbalance (534:1). A
    # stratified split is mandatory to ensure even the rarest class
    # (Worms, 174 rows total) is represented proportionally in both the
    # train and test sets.
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    log.append(
        f"Performed stratified train/test split "
        f"(test_size={TEST_SIZE}, random_state={RANDOM_STATE}). "
        f"Train shape: {X_train.shape}, Test shape: {X_test.shape}."
    )

    # Explicitly note: we deliberately did NOT deduplicate rows before this
    # split. EDA section 3 showed duplicate rate varies from 3.45% (Worms)
    # to 96.48% (Generic) across classes. Removing duplicates would:
    #   (a) disproportionately shrink already-tiny minority classes further
    #       if any of their rows happen to be duplicates, and
    #   (b) likely remove genuine repeated-attack-signature rows rather than
    #       erroneous data entry, since network attacks (e.g. generic/
    #       automated attacks) often produce near-identical flow records.
    # This decision should be explicitly justified in the report's
    # empirical process section.
    log.append(
        "No deduplication performed: duplicate rate is highly class-"
        "dependent (3.45% for Worms up to 96.48% for Generic per EDA "
        "section 3), so deduplication would disproportionately harm "
        "minority classes and likely discard genuine repeated attack "
        "signal rather than erroneous rows."
    )

    return X_train, X_test, y_train, y_test


def report_class_balance_after_split(y_train, y_test, log):
    train_counts = y_train.value_counts(normalize=True).sort_index() * 100
    test_counts = y_test.value_counts(normalize=True).sort_index() * 100
    table = pd.DataFrame({
        "train_pct": train_counts.round(3),
        "test_pct": test_counts.round(3),
    })
    table["attack_name_guess"] = [
        ATTACK_CAT_NAME_MAP.get(i, "?") for i in table.index
    ]
    log.append("\nClass balance check after stratified split (should match "
               "closely between train_pct and test_pct):")
    log.append(table.to_string())
    return table


def save_outputs(X_train, X_test, y_train, y_test, log, balance_table):
    X_train.to_csv(os.path.join(OUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUT_DIR, "y_test.csv"), index=False)

    balance_table.to_csv(os.path.join(OUT_DIR, "class_balance_after_split.csv"))

    # Save the exact feature column list used, since one-hot encoding
    # produces a specific set of columns that downstream scripts (training,
    # evaluation) must align to (e.g. when encoding new/unseen data).
    with open(os.path.join(OUT_DIR, "feature_columns.json"), "w") as f:
        json.dump(list(X_train.columns), f, indent=2)

    with open(os.path.join(OUT_DIR, "preprocessing_report.txt"), "w", encoding="utf-8") as f:
        f.write("PREPROCESSING PIPELINE REPORT\n")
        f.write("=" * 80 + "\n\n")
        for line in log:
            f.write(line + "\n\n")


def main():
    ensure_dirs()
    log = []

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Expected at data/networkTraffic.csv")
        return

    df = load_data()
    log.append(f"Loaded raw data: shape={df.shape}")

    df = drop_irrelevant_columns(df, log)
    df = fix_is_ftp_login(df, log)
    df = group_rare_proto(df, log)
    df = handle_service_placeholder(df, log)
    df = handle_state_anomaly(df, log)
    df = encode_categoricals(df, log)

    X_train, X_test, y_train, y_test = split_data(df, log)
    balance_table = report_class_balance_after_split(y_train, y_test, log)

    save_outputs(X_train, X_test, y_train, y_test, log, balance_table)

    print("\n".join(log))
    print(f"\nProcessed data written to: {OUT_DIR}")


if __name__ == "__main__":
    main()