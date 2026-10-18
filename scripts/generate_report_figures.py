"""
Generates all report-ready figures for Assignment 4 from the CSV outputs
already produced by scripts/run_eda.py and scripts/train_ensemble.py.

This script does NOT retrain or reprocess anything - it is a pure
visualization layer over existing results. Re-run it any time you want to
regenerate figures with updated styling without re-running the expensive
training pipeline.

Expects the following to already exist (produced by earlier scripts):
  data/processed/... (not used directly here)
  results/tables/target_class_distribution.csv
  results/tables/duplicates_by_class.csv
  results/tables/validation_curve_rf_n_estimators.csv
  results/tables/validation_curve_rf_max_depth.csv
  results/tables/validation_curve_gbm_learning_rate.csv
  results/tables/validation_curve_gbm_max_iter.csv
  results/tables/class_weight_comparison.csv
  results/tables/cv_results_rf.csv
  results/tables/cv_results_gbm.csv
  results/tables/test_set_classification_report_rf.csv
  results/tables/test_set_classification_report_gbm.csv
  results/tables/feature_importance_rf.csv
  results/tables/feature_importance_gbm.csv
  results/tables/model_comparison_summary.csv

Outputs all figures to results/figures/report/ (kept separate from the
exploratory figures already produced during EDA/training, so these can be
dropped straight into the report without sifting through intermediate plots).

Usage:
    python scripts/generate_report_figures.py
"""

import os
import ast

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

TABLE_DIR = os.path.join(REPO_ROOT, "results", "tables")
OUT_DIR = os.path.join(REPO_ROOT, "results", "figures", "report")

# Consistent, report-friendly styling: same font size for figure text as
# body text (per writing rule 25b), muted but distinguishable colors.
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
RF_COLOR = "#4C72B0"
GBM_COLOR = "#DD8452"

ATTACK_CAT_NAME_MAP = {
    0: "Normal", 9: "Generic", 4: "Exploits", 6: "Fuzzers", 3: "DoS",
    1: "Reconnaissance", 5: "Analysis", 2: "Backdoor", 8: "Shellcode", 7: "Worms",
}

CLASS_ORDER = list(ATTACK_CAT_NAME_MAP.values())


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def savefig(fig, name):
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 1. Class distribution (with readable attack names)
# ---------------------------------------------------------------------------
def plot_class_distribution():
    path = os.path.join(TABLE_DIR, "target_class_distribution.csv")
    if not os.path.exists(path):
        print(f"Skipping class distribution plot: {path} not found.")
        return

    df = pd.read_csv(path, index_col=0)
    df.index = [ATTACK_CAT_NAME_MAP.get(int(i), str(i)) for i in df.index]
    df = df.reindex(CLASS_ORDER)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df.index, df["count"], color=RF_COLOR)
    ax.set_ylabel("Number of instances")
    ax.set_xlabel("Attack category")
    ax.set_yscale("log")
    plt.xticks(rotation=35, ha="right")
    savefig(fig, "01_class_distribution")


# ---------------------------------------------------------------------------
# 2. Duplicate rate by class (ties EDA finding to class imbalance discussion)
# ---------------------------------------------------------------------------
def plot_duplicates_by_class():
    path = os.path.join(TABLE_DIR, "duplicates_by_class.csv")
    if not os.path.exists(path):
        print(f"Skipping duplicate rate plot: {path} not found.")
        return

    df = pd.read_csv(path, index_col=0)
    df.index = [ATTACK_CAT_NAME_MAP.get(int(i), str(i)) for i in df.index]
    df = df.reindex(CLASS_ORDER)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df.index, df["duplicate_rate_pct"], color="#8172B2")
    ax.set_ylabel("Duplicate rows (%)")
    ax.set_xlabel("Attack category")
    plt.xticks(rotation=35, ha="right")
    savefig(fig, "02_duplicate_rate_by_class")


# ---------------------------------------------------------------------------
# 3. Validation curves (RF: n_estimators, max_depth; GBM: learning_rate, max_iter)
# ---------------------------------------------------------------------------
def plot_validation_curve(csv_name, xlabel, out_name, categorical_x=False):
    path = os.path.join(TABLE_DIR, f"validation_curve_{csv_name}.csv")
    if not os.path.exists(path):
        print(f"Skipping validation curve {csv_name}: {path} not found.")
        return

    df = pd.read_csv(path)
    x_col = df.columns[0]
    x_values = df[x_col].astype(str) if categorical_x else df[x_col]
    x_plot = range(len(df)) if categorical_x else x_values

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(x_plot, df["train_macro_f1_mean"], "o-", label="Training", color=RF_COLOR)
    ax.fill_between(
        x_plot,
        df["train_macro_f1_mean"] - df["train_macro_f1_std"],
        df["train_macro_f1_mean"] + df["train_macro_f1_std"],
        alpha=0.2, color=RF_COLOR,
    )
    ax.plot(x_plot, df["val_macro_f1_mean"], "o-", label="Validation", color=GBM_COLOR)
    ax.fill_between(
        x_plot,
        df["val_macro_f1_mean"] - df["val_macro_f1_std"],
        df["val_macro_f1_mean"] + df["val_macro_f1_std"],
        alpha=0.2, color=GBM_COLOR,
    )
    if categorical_x:
        ax.set_xticks(list(x_plot))
        ax.set_xticklabels(x_values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Macro-averaged F1-score")
    ax.legend()
    savefig(fig, out_name)


# ---------------------------------------------------------------------------
# 4. Class-weight ablation (Q4)
# ---------------------------------------------------------------------------
def plot_class_weight_ablation():
    path = os.path.join(TABLE_DIR, "class_weight_comparison.csv")
    if not os.path.exists(path):
        print(f"Skipping class weight ablation plot: {path} not found.")
        return

    df = pd.read_csv(path, index_col=0)
    fold_rows = [r for r in df.index if r not in ("mean", "std")]
    fold_data = df.loc[fold_rows].astype(float)

    means = df.loc["mean"].astype(float)
    stds = df.loc["std"].astype(float)

    labels = ["Unweighted", "Balanced"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.bar(x, [means["None"], means["balanced"]],
           yerr=[stds["None"], stds["balanced"]],
           color=[RF_COLOR, GBM_COLOR], capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean macro-averaged F1-score (5-fold cross-validation)")
    savefig(fig, "05_class_weight_ablation")


# ---------------------------------------------------------------------------
# 5. CV best-score vs held-out test macro-F1 (illustrates the leakage gap)
# ---------------------------------------------------------------------------
def plot_cv_vs_test_gap():
    cv_rf_path = os.path.join(TABLE_DIR, "cv_results_rf.csv")
    cv_gbm_path = os.path.join(TABLE_DIR, "cv_results_gbm.csv")
    comp_path = os.path.join(TABLE_DIR, "model_comparison_summary.csv")

    if not all(os.path.exists(p) for p in (cv_rf_path, cv_gbm_path, comp_path)):
        print("Skipping CV-vs-test gap plot: one or more required files not found.")
        return

    cv_rf = pd.read_csv(cv_rf_path)
    cv_gbm = pd.read_csv(cv_gbm_path)
    comp = pd.read_csv(comp_path)

    best_cv_rf = cv_rf.sort_values("rank_test_score").iloc[0]["mean_test_score"]
    best_cv_gbm = cv_gbm.sort_values("rank_test_score").iloc[0]["mean_test_score"]

    test_rf = comp.loc[comp["model"] == "rf", "test_macro_f1"].values[0]
    test_gbm = comp.loc[comp["model"] == "gbm", "test_macro_f1"].values[0]

    labels = ["Random forest", "Gradient boosting"]
    cv_scores = [best_cv_rf, best_cv_gbm]
    test_scores = [test_rf, test_gbm]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x - width / 2, cv_scores, width, label="Best cross-validation score", color=RF_COLOR)
    ax.bar(x + width / 2, test_scores, width, label="Held-out test score", color=GBM_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Macro-averaged F1-score")
    ax.legend()
    savefig(fig, "06_cv_vs_test_gap")


# ---------------------------------------------------------------------------
# 6. Per-class F1 comparison (RF vs GBM) - highlights Backdoor/Analysis weakness
# ---------------------------------------------------------------------------
def plot_per_class_f1_comparison():
    rf_path = os.path.join(TABLE_DIR, "test_set_classification_report_rf.csv")
    gbm_path = os.path.join(TABLE_DIR, "test_set_classification_report_gbm.csv")
    if not (os.path.exists(rf_path) and os.path.exists(gbm_path)):
        print("Skipping per-class F1 comparison: classification reports not found.")
        return

    rf_df = pd.read_csv(rf_path, index_col=0)
    gbm_df = pd.read_csv(gbm_path, index_col=0)

    class_rows = [c for c in CLASS_ORDER if c in rf_df.index]

    rf_f1 = rf_df.loc[class_rows, "f1-score"]
    gbm_f1 = gbm_df.loc[class_rows, "f1-score"]

    x = np.arange(len(class_rows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, rf_f1, width, label="Random forest", color=RF_COLOR)
    ax.bar(x + width / 2, gbm_f1, width, label="Gradient boosting", color=GBM_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(class_rows, rotation=35, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_xlabel("Attack category")
    ax.legend()
    savefig(fig, "07_per_class_f1_comparison")


# ---------------------------------------------------------------------------
# 7. Precision/recall breakdown specifically for Backdoor and Analysis
# ---------------------------------------------------------------------------
def plot_backdoor_analysis_precision_recall():
    rf_path = os.path.join(TABLE_DIR, "test_set_classification_report_rf.csv")
    gbm_path = os.path.join(TABLE_DIR, "test_set_classification_report_gbm.csv")
    if not (os.path.exists(rf_path) and os.path.exists(gbm_path)):
        print("Skipping Backdoor/Analysis precision-recall plot: reports not found.")
        return

    rf_df = pd.read_csv(rf_path, index_col=0)
    gbm_df = pd.read_csv(gbm_path, index_col=0)

    focus_classes = ["Backdoor", "Analysis"]
    metrics = ["precision", "recall"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    for ax, cls in zip(axes, focus_classes):
        rf_vals = rf_df.loc[cls, metrics].astype(float)
        gbm_vals = gbm_df.loc[cls, metrics].astype(float)

        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width / 2, rf_vals, width, label="Random forest", color=RF_COLOR)
        ax.bar(x + width / 2, gbm_vals, width, label="Gradient boosting", color=GBM_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(["Precision", "Recall"])
        ax.set_title(cls)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Score")
    axes[0].legend()
    savefig(fig, "08_backdoor_analysis_precision_recall")


# ---------------------------------------------------------------------------
# 8. Feature importance comparison (top 10 overlap between RF and GBM)
# ---------------------------------------------------------------------------
def plot_feature_importance_comparison():
    rf_path = os.path.join(TABLE_DIR, "feature_importance_rf.csv")
    gbm_path = os.path.join(TABLE_DIR, "feature_importance_gbm.csv")
    if not (os.path.exists(rf_path) and os.path.exists(gbm_path)):
        print("Skipping feature importance comparison: files not found.")
        return

    rf_df = pd.read_csv(rf_path, index_col=0).iloc[:, 0]
    gbm_df = pd.read_csv(gbm_path, index_col=0).iloc[:, 0]

    rf_top = rf_df.sort_values(ascending=False).head(10)
    gbm_top = gbm_df.sort_values(ascending=False).head(10)

    union_features = list(dict.fromkeys(list(rf_top.index) + list(gbm_top.index)))

    rf_vals = rf_df.reindex(union_features).fillna(0)
    gbm_vals = gbm_df.reindex(union_features).fillna(0)

    # Normalize each model's importances to sum to 1 across the union so the
    # two differently-scaled importance measures (impurity-based for RF,
    # permutation-based for GBM) are visually comparable.
    rf_vals_norm = rf_vals / rf_vals.sum()
    gbm_vals_norm = gbm_vals / gbm_vals.sum()

    order = (rf_vals_norm + gbm_vals_norm).sort_values(ascending=True).index
    rf_vals_norm = rf_vals_norm.reindex(order)
    gbm_vals_norm = gbm_vals_norm.reindex(order)

    y = np.arange(len(order))
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y - height / 2, rf_vals_norm, height, label="Random forest", color=RF_COLOR)
    ax.barh(y + height / 2, gbm_vals_norm, height, label="Gradient boosting", color=GBM_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Normalized feature importance")
    ax.legend()
    savefig(fig, "09_feature_importance_comparison")


# ---------------------------------------------------------------------------
# 9. Overall model comparison summary (macro-F1, weighted-F1, accuracy)
# ---------------------------------------------------------------------------
def plot_overall_model_comparison():
    path = os.path.join(TABLE_DIR, "model_comparison_summary.csv")
    if not os.path.exists(path):
        print(f"Skipping overall model comparison plot: {path} not found.")
        return

    df = pd.read_csv(path)
    df["model"] = df["model"].map({"rf": "Random forest", "gbm": "Gradient boosting"})

    metrics = ["test_macro_f1", "test_weighted_f1", "test_accuracy"]
    metric_labels = ["Macro F1-score", "Weighted F1-score", "Accuracy"]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (model_name, color) in enumerate(
        zip(df["model"], [RF_COLOR, GBM_COLOR])
    ):
        values = df.loc[df["model"] == model_name, metrics].values.flatten()
        offset = (i - 0.5) * width
        ax.bar(x + offset, values, width, label=model_name, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    savefig(fig, "10_overall_model_comparison")


# ---------------------------------------------------------------------------
# 10. Computational cost comparison (tuning time, prediction time)
# ---------------------------------------------------------------------------
def plot_computational_cost():
    path = os.path.join(TABLE_DIR, "model_comparison_summary.csv")
    if not os.path.exists(path):
        print(f"Skipping computational cost plot: {path} not found.")
        return

    df = pd.read_csv(path)
    df["model"] = df["model"].map({"rf": "Random forest", "gbm": "Gradient boosting"})

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].bar(df["model"], df["tuning_time_s"] / 60, color=[RF_COLOR, GBM_COLOR])
    axes[0].set_ylabel("Tuning time (minutes)")

    axes[1].bar(df["model"], df["predict_time_s"], color=[RF_COLOR, GBM_COLOR])
    axes[1].set_ylabel("Test set prediction time (seconds)")

    savefig(fig, "11_computational_cost_comparison")


def main():
    ensure_dirs()

    plot_class_distribution()
    plot_duplicates_by_class()

    plot_validation_curve("rf_n_estimators", "Number of trees",
                           "03a_validation_curve_rf_n_estimators")
    plot_validation_curve("rf_max_depth", "Maximum tree depth",
                           "03b_validation_curve_rf_max_depth", categorical_x=True)
    plot_validation_curve("gbm_learning_rate", "Learning rate",
                           "04a_validation_curve_gbm_learning_rate")
    plot_validation_curve("gbm_max_iter", "Maximum number of boosting iterations",
                           "04b_validation_curve_gbm_max_iter")

    plot_class_weight_ablation()
    plot_cv_vs_test_gap()
    plot_per_class_f1_comparison()
    plot_backdoor_analysis_precision_recall()
    plot_feature_importance_comparison()
    plot_overall_model_comparison()
    plot_computational_cost()

    print(f"\nAll report figures written to: {OUT_DIR}")


if __name__ == "__main__":
    main()