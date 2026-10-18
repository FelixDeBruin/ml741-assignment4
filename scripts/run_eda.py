"""
Full Exploratory Data Analysis (EDA) for the network traffic / cyberattack
dataset used in Assignment 4 (ensemble learning option).

This script is read-only with respect to the raw data: it does not modify
data/networkTraffic.csv. It writes:
  - figures to results/figures/
  - summary tables (CSV) to results/tables/
  - a single consolidated text log to results/eda_summary.txt

Run scripts/inspect_dataset.py first if you want a quick, unstructured look
at the data before running this full pipeline.

Usage:
    python scripts/run_eda.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(REPO_ROOT, "data", "networkTraffic.csv")
FEATURES_PATH = os.path.join(REPO_ROOT, "data", "features.csv")

FIG_DIR = os.path.join(REPO_ROOT, "results", "figures")
TABLE_DIR = os.path.join(REPO_ROOT, "results", "tables")
SUMMARY_PATH = os.path.join(REPO_ROOT, "results", "eda_summary.txt")

TARGET = "attack_cat"
ID_COL = "id"

# Known nominal/categorical columns based on features.csv + header inspection.
CATEGORICAL_COLS = ["proto", "service", "state"]

# Binary / flag-like columns (treated separately from continuous numerics
# for distribution plots, since histograms of 0/1 flags are not informative
# in the same way as continuous features).
BINARY_COLS = [
    "is_ftp_login",
    "is_sm_ips_ports",
]

# Columns present in the data but not described in features.csv.
# id: row index, safe to drop.
# rate: appears derived (likely (spkts+dpkts)/dur or similar) - undocumented,
# flag this explicitly in the report rather than silently using it.
UNDOCUMENTED_COLS = ["id", "rate"]

# attack_cat is already label-encoded as integers 0-9 with NO mapping back to
# the original attack category names (e.g. Normal, Fuzzers, DoS, Generic...)
# provided anywhere in the dataset or features.csv. This script therefore
# treats attack_cat purely as a categorical label (not an ordinal/continuous
# number) throughout - note this limitation explicitly in your report.
N_CLASSES = 10

# High-cardinality categorical handling: proto has 133 unique values, which
# is impractical to one-hot encode directly or to plot in full. We only plot
# the top-N most frequent categories and group the remainder as "other" for
# visualization purposes only (the raw column is left untouched for modeling
# decisions, which you will make separately in preprocessing).
TOP_N_CARDINALITY = 15

sns.set_theme(style="whitegrid", context="talk")


class Logger:
    """Tiny helper to print to stdout and append to a summary file."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(self.path, "w").close()

    def section(self, title):
        line = "\n" + "=" * 80 + f"\n{title}\n" + "=" * 80
        self._write(line)

    def write(self, obj):
        self._write(str(obj))

    def _write(self, text):
        print(text)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    features = pd.read_csv(FEATURES_PATH)
    return df, features


def get_object_cols(df):
    """Robust to pandas 2/3 string dtype migration (avoids Pandas4Warning)."""
    try:
        return df.select_dtypes(include="str").columns
    except TypeError:
        return df.select_dtypes(include="object").columns


def basic_overview(df, log):
    log.section("1. BASIC OVERVIEW")
    log.write(f"Shape: {df.shape}")
    log.write("\nDtypes:")
    log.write(df.dtypes)

    log.write(f"\nUndocumented columns (not in features.csv): {UNDOCUMENTED_COLS}")
    log.write("  'id' is a row index -> drop before modeling.")
    log.write("  'rate' appears derived from packet/byte counts over duration "
               "-> inspect correlation with sbytes/dbytes/spkts/dpkts/dur "
               "before deciding whether to keep (possible redundancy).")

    dtype_table = df.dtypes.astype(str).reset_index()
    dtype_table.columns = ["column", "dtype"]
    dtype_table.to_csv(os.path.join(TABLE_DIR, "column_dtypes.csv"), index=False)


def missingness_analysis(df, log):
    log.section("2. MISSINGNESS / PLACEHOLDER ANALYSIS")

    nan_counts = df.isnull().sum()
    nan_pct = (nan_counts / len(df) * 100).round(2)

    placeholder_counts = {}
    for c in get_object_cols(df):
        n = (df[c] == "?").sum()
        if n > 0:
            placeholder_counts[c] = n

    log.write("NaN counts (non-zero only):")
    log.write(nan_counts[nan_counts > 0] if nan_counts.sum() > 0 else "  None")

    log.write("\n'?' placeholder counts (non-zero only):")
    if placeholder_counts:
        for c, n in placeholder_counts.items():
            log.write(f"  {c}: {n} ({n / len(df):.2%})")
            if n / len(df) > 0.3:
                log.write(f"  NOTE: {c} has >30% placeholder values - this is "
                          f"too large to drop rows; treat '?' as an explicit "
                          f"'unknown' category rather than imputing.")
    else:
        log.write("  None")

    missing_summary = pd.DataFrame(
        {
            "column": nan_counts.index,
            "nan_count": nan_counts.values,
            "nan_pct": nan_pct.values,
        }
    )
    missing_summary["placeholder_q_count"] = missing_summary["column"].map(
        placeholder_counts
    ).fillna(0).astype(int)
    missing_summary.to_csv(os.path.join(TABLE_DIR, "missingness_summary.csv"), index=False)

    if placeholder_counts:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(placeholder_counts.keys(), placeholder_counts.values(), color="#4C72B0")
        ax.set_ylabel("Count of '?' placeholder")
        ax.set_title("Placeholder ('?') value counts by column")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "missingness_placeholder_counts.png"), dpi=150)
        plt.close(fig)

    # service '?' vs target: is missingness informative of attack type?
    if "service" in df.columns:
        service_missing = (df["service"] == "?").astype(int)
        ct = pd.crosstab(df[TARGET], service_missing, normalize="index") * 100
        ct.columns = ["service_known_pct", "service_unknown_pct"]
        ct.to_csv(os.path.join(TABLE_DIR, "service_missingness_by_class.csv"))
        log.write("\n'service' missingness rate (%) by attack_cat class "
                  "(checks whether missingness itself is predictive):")
        log.write(ct)


def duplicates_and_constants(df, log):
    log.section("3. DUPLICATES AND CONSTANT COLUMNS")

    cols_no_id = [c for c in df.columns if c != ID_COL]
    dup_mask = df.duplicated(subset=cols_no_id, keep=False)
    n_dupes = df.duplicated(subset=cols_no_id).sum()
    log.write(f"Duplicate rows (excluding id, first occurrence kept): {n_dupes} "
              f"({n_dupes / len(df):.2%})")
    log.write("NOTE: >30% duplication is substantial. This may reflect "
              "repeated/replayed traffic patterns (plausible for network data, "
              "especially for automated attacks) rather than a data error, but "
              "must be investigated and justified before deciding whether to "
              "deduplicate prior to modeling (deduplicating could remove "
              "legitimate repeated attack signatures and bias class balance).")

    # Are duplicates concentrated in specific classes?
    dup_by_class = df.loc[dup_mask].groupby(TARGET).size()
    total_by_class = df.groupby(TARGET).size()
    dup_rate_by_class = (dup_by_class / total_by_class * 100).round(2).fillna(0)
    dup_summary = pd.DataFrame({
        "total_rows": total_by_class,
        "rows_involved_in_duplicates": dup_by_class.reindex(total_by_class.index).fillna(0).astype(int),
        "duplicate_rate_pct": dup_rate_by_class,
    })
    log.write("\nDuplicate involvement rate (%) by attack_cat class:")
    log.write(dup_summary)
    dup_summary.to_csv(os.path.join(TABLE_DIR, "duplicates_by_class.csv"))

    fig, ax = plt.subplots(figsize=(9, 5))
    dup_summary["duplicate_rate_pct"].plot(kind="bar", ax=ax, color="#DD8452")
    ax.set_ylabel("% of rows in class that are duplicates")
    ax.set_xlabel("attack_cat")
    ax.set_title("Duplicate row rate by attack_cat class")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "duplicates_by_class.png"), dpi=150)
    plt.close(fig)

    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    log.write(f"\nConstant / single-value columns: {constant_cols if constant_cols else 'None'}")

    near_constant = []
    for c in df.columns:
        if c in (ID_COL, TARGET):
            continue
        top_freq = df[c].value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq > 0.99:
            near_constant.append((c, round(top_freq, 4)))
    log.write(f"Near-constant columns (>99% single value): {near_constant if near_constant else 'None'}")


def target_analysis(df, log):
    log.section("4. TARGET VARIABLE (attack_cat) ANALYSIS")

    log.write(f"dtype: {df[TARGET].dtype}")
    log.write(f"unique values: {sorted(df[TARGET].unique())} ({df[TARGET].nunique()} classes)")
    log.write("NOTE: attack_cat is pre-encoded as integers 0-9 with NO mapping "
              "file provided linking codes to attack category names (e.g. "
              "Normal, Fuzzers, DoS, Generic, Reconnaissance, etc.). This "
              "script therefore treats attack_cat strictly as a nominal "
              "categorical label (not ordinal) in all plots/analysis below. "
              "State this limitation explicitly in your report, and consider "
              "cross-referencing the public UNSW-NB15 documentation if you "
              "wish to recover semantic class names.")

    counts = df[TARGET].value_counts().sort_index()
    pct = (counts / len(df) * 100).round(2)
    table = pd.DataFrame({"count": counts, "pct": pct})
    log.write("\nClass distribution:")
    log.write(table)
    table.to_csv(os.path.join(TABLE_DIR, "target_class_distribution.csv"))

    imbalance_ratio = counts.max() / counts.min()
    log.write(f"\nImbalance ratio (majority/minority): {imbalance_ratio:.1f}:1")
    log.write("NOTE: severe imbalance (500+:1). Favor macro-averaged F1 (or "
              "per-class F1) over plain accuracy, use stratified train/test "
              "splits and stratified CV folds, and consider class_weight "
              "or resampling strategies during tuning.")

    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax, color="#C44E52")
    ax.set_ylabel("Count")
    ax.set_xlabel("attack_cat (encoded, no name mapping available)")
    ax.set_title("Class distribution of attack_cat")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "target_class_distribution.png"), dpi=150)
    plt.close(fig)


def categorical_analysis(df, log):
    log.section("5. CATEGORICAL FEATURE ANALYSIS (proto, service, state)")

    for c in CATEGORICAL_COLS:
        if c not in df.columns:
            continue
        n_unique = df[c].nunique()
        log.write(f"\n--- {c} ({n_unique} unique values) ---")
        vc = df[c].value_counts()
        log.write(vc.head(20))
        vc.to_csv(os.path.join(TABLE_DIR, f"cardinality_{c}.csv"))

        if n_unique > TOP_N_CARDINALITY:
            log.write(f"  NOTE: {c} has high cardinality ({n_unique} unique "
                      f"values). Plot below shows only top {TOP_N_CARDINALITY} "
                      f"categories grouped with an 'other' bucket for the "
                      f"remainder. For modeling, consider grouping rare "
                      f"categories (e.g. <0.1% frequency) into 'other' before "
                      f"one-hot encoding to avoid extremely sparse, high-"
                      f"dimensional features.")
            top = vc.head(TOP_N_CARDINALITY)
            other_count = vc.iloc[TOP_N_CARDINALITY:].sum()
            plot_series = pd.concat([top, pd.Series({"other": other_count})])
        else:
            plot_series = vc

        fig, ax = plt.subplots(figsize=(10, 5))
        plot_series.plot(kind="bar", ax=ax, color="#55A868")
        ax.set_title(f"Category counts: {c}" +
                     (f" (top {TOP_N_CARDINALITY} + other)" if n_unique > TOP_N_CARDINALITY else ""))
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        plt.xticks(rotation=40, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"categorical_{c}_counts.png"), dpi=150)
        plt.close(fig)

    # state column contains an unexpected 'no' value alongside standard TCP
    # states (FIN, CON, INT, REQ, RST, etc.) - flag explicitly.
    if "state" in df.columns:
        standard_states = {"ACC", "CLO", "CON", "ECO", "ECR", "FIN", "INT",
                            "MAS", "PAR", "REQ", "RST", "TST", "TXD", "URH", "URN"}
        observed = set(df["state"].unique())
        unexpected = observed - standard_states
        if unexpected:
            log.write(f"\nNOTE: 'state' contains value(s) not in the documented "
                      f"state list from features.csv: {unexpected}. Verify "
                      f"these are not data entry artifacts before encoding.")

    # Cross-tab: categorical features vs target, useful to see whether
    # certain protocols/services/states are strongly associated with
    # particular attack categories. attack_cat treated as categorical columns.
    for c in CATEGORICAL_COLS:
        if c not in df.columns:
            continue
        ct = pd.crosstab(df[c], df[TARGET])
        ct.to_csv(os.path.join(TABLE_DIR, f"crosstab_{c}_vs_target.csv"))

        # Normalized version (row-wise %) is often more interpretable for
        # high-cardinality columns like proto.
        ct_norm = pd.crosstab(df[c], df[TARGET], normalize="index").round(4) * 100
        ct_norm.to_csv(os.path.join(TABLE_DIR, f"crosstab_{c}_vs_target_rowpct.csv"))


def numeric_distribution_analysis(df, log):
    log.section("6. NUMERIC FEATURE DISTRIBUTIONS")

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in (ID_COL, TARGET) and c not in BINARY_COLS
    ]

    desc = df[numeric_cols].describe().T
    desc["skew"] = df[numeric_cols].skew()
    desc["kurtosis"] = df[numeric_cols].kurtosis()
    log.write(desc)
    desc.to_csv(os.path.join(TABLE_DIR, "numeric_summary_stats.csv"))

    highly_skewed = desc[desc["skew"].abs() > 5].index.tolist()
    log.write(f"\nHighly skewed numeric columns (|skew| > 5): {highly_skewed}")
    log.write("NOTE: tree-based ensembles (Random Forest, Gradient Boosting) "
              "are invariant to monotonic transformations of individual "
              "features, so skew alone is not a strong justification for "
              "transforming features ahead of ensemble modeling - mention "
              "this explicitly in your report's preprocessing justification.")

    n_cols = 4
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i, c in enumerate(numeric_cols):
        ax = axes[i]
        data = df[c].dropna()
        ax.hist(data, bins=40, color="#4C72B0")
        ax.set_title(c, fontsize=10)
        ax.set_yscale("log")
    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "numeric_histograms_grid.png"), dpi=130)
    plt.close(fig)

    # rate vs derived candidates - check redundancy with other packet/byte
    # fields since 'rate' is undocumented in features.csv.
    candidates = [c for c in ["spkts", "dpkts", "sbytes", "dbytes", "dur"] if c in df.columns]
    if "rate" in df.columns and candidates:
        rate_corr = df[["rate"] + candidates].corr()["rate"].drop("rate")
        log.write("\nCorrelation of undocumented 'rate' column with packet/byte/duration fields:")
        log.write(rate_corr.sort_values(key=lambda s: s.abs(), ascending=False))


def binary_flag_analysis(df, log):
    log.section("7. BINARY / FLAG FEATURE ANALYSIS")
    for c in BINARY_COLS:
        if c not in df.columns:
            continue
        vc = df[c].value_counts()
        log.write(f"\n--- {c} ---")
        log.write(vc)
        if set(df[c].unique()) - {0, 1}:
            log.write(f"  NOTE: {c} is documented as Binary but contains "
                      f"values outside {{0,1}}: {sorted(df[c].unique())}. "
                      f"Verify this is expected (e.g. ct_ftp_cmd-style counts "
                      f"mislabeled as binary) before treating it as a flag.")


def correlation_analysis(df, log):
    log.section("8. CORRELATION ANALYSIS (numeric features)")

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in (ID_COL,)
    ]
    corr = df[numeric_cols].corr(numeric_only=True)
    corr.to_csv(os.path.join(TABLE_DIR, "numeric_feature_correlation.csv"))

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, ax=ax,
                cbar_kws={"shrink": 0.6})
    ax.set_title("Correlation matrix: numeric features + target")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"), dpi=130)
    plt.close(fig)

    threshold = 0.9
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notnull(v) and abs(v) >= threshold:
                pairs.append((cols[i], cols[j], round(v, 3)))
    pairs_df = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "correlation"])
    pairs_df = pairs_df.sort_values("correlation", key=lambda s: s.abs(), ascending=False)
    pairs_df.to_csv(os.path.join(TABLE_DIR, "highly_correlated_pairs.csv"), index=False)
    log.write(f"\nHighly correlated feature pairs (|r| >= {threshold}):")
    log.write(pairs_df)

    # attack_cat is nominal despite being int-encoded, so a raw Pearson
    # correlation with it is of limited validity (no ordinal meaning between
    # classes). Reported here only as a rough, heavily-caveated univariate
    # signal check - do not over-interpret in the report.
    target_corr = corr[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)
    target_corr.to_csv(os.path.join(TABLE_DIR, "feature_target_correlation_CAVEAT.csv"))
    log.write("\nTop 15 features by |Pearson correlation| with attack_cat "
              "(CAVEAT: attack_cat is nominal, not ordinal - this is a rough "
              "signal check only, not a rigorous association measure):")
    log.write(target_corr.head(15))


def feature_vs_target_boxplots(df, log):
    log.section("9. TOP FEATURES vs TARGET (boxplots)")

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in (ID_COL, TARGET) and c not in BINARY_COLS
    ]

    corr = df[numeric_cols + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
    top_features = corr.abs().sort_values(ascending=False).head(6).index.tolist()

    log.write(f"Plotting boxplots for (selected via rough correlation signal): {top_features}")

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    for i, c in enumerate(top_features):
        ax = axes[i]
        sns.boxplot(x=TARGET, y=c, data=df, ax=ax, showfliers=False)
        ax.set_title(c)
        ax.set_xlabel("attack_cat")
        ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "top_features_vs_target_boxplots.png"), dpi=130)
    plt.close(fig)


def class_imbalance_table(df, log):
    log.section("10. CLASS IMBALANCE SUMMARY (for stratified split / class-weight decisions)")
    counts = df[TARGET].value_counts().sort_values(ascending=False)
    total = len(df)
    summary = pd.DataFrame({
        "class": counts.index,
        "count": counts.values,
        "pct_of_total": (counts.values / total * 100).round(3),
    })
    summary["ratio_vs_majority"] = (counts.max() / counts.values).round(2)
    log.write(summary)
    summary.to_csv(os.path.join(TABLE_DIR, "class_imbalance_summary.csv"), index=False)


def main():
    ensure_dirs()
    log = Logger(SUMMARY_PATH)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Expected at data/networkTraffic.csv")
        sys.exit(1)

    df, features = load_data()

    basic_overview(df, log)
    missingness_analysis(df, log)
    duplicates_and_constants(df, log)
    target_analysis(df, log)
    categorical_analysis(df, log)
    numeric_distribution_analysis(df, log)
    binary_flag_analysis(df, log)
    correlation_analysis(df, log)
    feature_vs_target_boxplots(df, log)
    class_imbalance_table(df, log)

    log.section("EDA COMPLETE")
    log.write(f"Figures written to: {FIG_DIR}")
    log.write(f"Tables written to: {TABLE_DIR}")
    log.write(f"Full log written to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()