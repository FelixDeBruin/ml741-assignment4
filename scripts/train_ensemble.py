"""
Ensemble training and tuning pipeline for Assignment 4 (ensemble learning
option). Trains and tunes two ensemble paradigms on the preprocessed
UNSW-NB15-style dataset:

  - RandomForestClassifier         (bagging / parallel ensemble)
  - HistGradientBoostingClassifier (boosting / sequential ensemble,
                                     histogram-based for speed on large data)

NOTE ON HistGradientBoostingClassifier vs GradientBoostingClassifier:
We switched from the standard GradientBoostingClassifier to the histogram-
based variant because standard GBM trains strictly sequentially with no
multi-core speedup per fit, making it prohibitively slow on this dataset
(206k rows). HistGradientBoostingClassifier bins continuous features and
uses a much faster split-finding algorithm (same family as LightGBM), while
remaining a legitimate sequential/boosting ensemble for the bagging-vs-
boosting comparison central to this assignment. This substitution and its
justification should be stated explicitly in your report's Implementation
section.

This directly operationalises the tuning questions set out in the project
plan:
  Q1: Does n_estimators reduce variance without overfitting? (RF)
  Q2: Does max_depth control the bias-variance tradeoff as ensemble theory
      predicts? (RF)
  Q3: Does the learning_rate / n_estimators tradeoff behave as boosting
      theory predicts? (HistGBM)
  Q4: Does class_weight='balanced' improve minority-class detection
      (macro-F1, per-class recall) relative to unweighted training? (RF)

Requires scripts/preprocessing.py to have already been run (reads from
data/processed/).

*** THIS IS A REDUCED-SCOPE SPEED-TEST CONFIGURATION ***
Grids and N_RANDOM_SEARCH_ITER have been deliberately shrunk to validate
that HistGradientBoostingClassifier runs fast enough before committing to
the full, more thorough search. See the "SPEED TEST" comments below for
exactly what to restore once this run completes quickly.

Outputs (under <repo_root>/results/):
  - tables/cv_results_rf.csv, cv_results_gbm.csv       (full CV search results)
  - tables/validation_curve_*.csv                      (data behind each plot)
  - figures/validation_curve_*.png                     (answers to Q1-Q4)
  - tables/test_set_classification_report_rf.csv
  - tables/test_set_classification_report_gbm.csv
  - tables/model_comparison_summary.csv                (final RF vs GBM table)
  - tables/feature_importance_rf.csv
  - tables/feature_importance_gbm.csv
  - figures/confusion_matrix_rf.png, confusion_matrix_gbm.png
  - figures/feature_importance_rf.png, feature_importance_gbm.png
  - models/best_rf.joblib, models/best_gbm.joblib
  - training_log.txt

Usage:
    python scripts/train_ensemble.py
"""

import os
import json
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    validation_curve,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
FIG_DIR = os.path.join(REPO_ROOT, "results", "figures")
TABLE_DIR = os.path.join(REPO_ROOT, "results", "tables")
MODEL_DIR = os.path.join(REPO_ROOT, "results", "models")
LOG_PATH = os.path.join(REPO_ROOT, "results", "training_log.txt")

RANDOM_STATE = 42

# SPEED TEST: reduced from 5 -> 3 folds. Restore to 5 once HistGBM speed is
# confirmed acceptable.
N_CV_FOLDS = 5

# SPEED TEST: reduced from 25 -> 10. Restore to 25 once confirmed fast.
N_RANDOM_SEARCH_ITER = 25

ATTACK_CAT_NAME_MAP = {
    0: "Normal", 9: "Generic", 4: "Exploits", 6: "Fuzzers", 3: "DoS",
    1: "Reconnaissance", 5: "Analysis", 2: "Backdoor", 8: "Shellcode", 7: "Worms",
}

sns.set_theme(style="whitegrid", context="talk")

macro_f1_scorer = make_scorer(f1_score, average="macro")


class Logger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(self.path, "w").close()

    def section(self, title):
        self._write("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)

    def write(self, obj):
        self._write(str(obj))

    def _write(self, text):
        print(text)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def ensure_dirs():
    for d in (FIG_DIR, TABLE_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)


def load_processed_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).iloc[:, 0]
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Q1 & Q2: Random Forest validation curves (n_estimators, max_depth)
# ---------------------------------------------------------------------------
def rf_validation_curves(X_train, y_train, log):
    log.section("Q1/Q2: RANDOM FOREST VALIDATION CURVES "
                "(n_estimators, max_depth)")

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    base_rf = RandomForestClassifier(
        random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
    )

    # SPEED TEST: reduced from [10,25,50,100,200,300] -> smaller set.
    # Restore full range once HistGBM speed is confirmed acceptable.
    n_estimators_range = [10,25,50,100,200,300]
    print(f"[RF] Starting n_estimators validation curve ({len(n_estimators_range)} values x {N_CV_FOLDS} folds)...")
    t0 = time.time()
    train_scores, val_scores = validation_curve(
        base_rf, X_train, y_train,
        param_name="n_estimators", param_range=n_estimators_range,
        cv=cv, scoring=macro_f1_scorer, n_jobs=-1,
    )
    elapsed = time.time() - t0
    log.write(f"n_estimators validation curve computed in {elapsed:.1f}s")

    _save_validation_curve(
        "rf_n_estimators", n_estimators_range, train_scores, val_scores,
        xlabel="n_estimators", title="Random Forest: n_estimators vs macro-F1",
        log=log,
    )

    # SPEED TEST: reduced range. Restore [2,4,6,8,12,16,20,None] later.
    max_depth_range = [2,4,6,8,12,16,20,None]
    max_depth_labels = [str(d) for d in max_depth_range]
    print(f"[RF] Starting max_depth validation curve ({len(max_depth_range)} values x {N_CV_FOLDS} folds)...")
    t0 = time.time()
    train_scores, val_scores = validation_curve(
        base_rf, X_train, y_train,
        param_name="max_depth", param_range=max_depth_range,
        cv=cv, scoring=macro_f1_scorer, n_jobs=-1,
    )
    elapsed = time.time() - t0
    log.write(f"max_depth validation curve computed in {elapsed:.1f}s")

    _save_validation_curve(
        "rf_max_depth", max_depth_labels, train_scores, val_scores,
        xlabel="max_depth", title="Random Forest: max_depth vs macro-F1",
        log=log, categorical_x=True,
    )


# ---------------------------------------------------------------------------
# Q3: HistGradientBoosting validation curves (learning_rate, n_estimators)
# ---------------------------------------------------------------------------
def gbm_validation_curves(X_train, y_train, log):
    log.section("Q3: HIST GRADIENT BOOSTING VALIDATION CURVES "
                "(learning_rate, max_iter)")

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # BUG FIX: HistGradientBoostingClassifier enables early_stopping=True by
    # default once training set size exceeds 10,000 rows. This silently
    # caps the effective number of boosting rounds regardless of the
    # max_iter value requested, which is why the max_iter validation curve
    # previously returned identical scores for max_iter=50/100/200 (all
    # three were being stopped early at the same point). We disable early
    # stopping here so max_iter varies as intended for this diagnostic
    # curve. For the final tuned model in tune_gradient_boosting(), we
    # leave early_stopping at its default (True), and treat that as an
    # implicit, built-in form of max_iter selection - this should be
    # explicitly justified in your report rather than grid-searching
    # max_iter exhaustively in the final search.
    base_gbm = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE, early_stopping=False
    )

    # SPEED TEST: reduced from 5 -> 3 values.
    learning_rate_range = [0.01, 0.05, 0.1, 0.2, 0.3]
    print(f"[HistGBM] Starting learning_rate validation curve ({len(learning_rate_range)} values x {N_CV_FOLDS} folds)...")
    t0 = time.time()
    train_scores, val_scores = validation_curve(
        base_gbm, X_train, y_train,
        param_name="learning_rate", param_range=learning_rate_range,
        cv=cv, scoring=macro_f1_scorer, n_jobs=-1,
    )
    elapsed = time.time() - t0
    log.write(f"learning_rate validation curve computed in {elapsed:.1f}s")

    _save_validation_curve(
        "gbm_learning_rate", learning_rate_range, train_scores, val_scores,
        xlabel="learning_rate", title="HistGradientBoosting: learning_rate vs macro-F1",
        log=log,
    )

    # SPEED TEST: reduced from [25,50,100,200] -> [50,100,200].
    # NOTE: with early_stopping=False, these will now train the FULL
    # max_iter rounds every time (no early exit), so this will be noticeably
    # slower per fit than the previous (buggy) run, but the scores will
    # actually be meaningful now - they should diverge by max_iter value.
    max_iter_range = [25, 50, 100, 200]
    print(f"[HistGBM] Starting max_iter validation curve ({len(max_iter_range)} values x {N_CV_FOLDS} folds)...")
    t0 = time.time()
    train_scores, val_scores = validation_curve(
        base_gbm, X_train, y_train,
        param_name="max_iter", param_range=max_iter_range,
        cv=cv, scoring=macro_f1_scorer, n_jobs=-1,
    )
    elapsed = time.time() - t0
    log.write(f"max_iter validation curve computed in {elapsed:.1f}s")

    _save_validation_curve(
        "gbm_max_iter", max_iter_range, train_scores, val_scores,
        xlabel="max_iter", title="HistGradientBoosting: max_iter vs macro-F1",
        log=log,
    )


def _save_validation_curve(name, x_values, train_scores, val_scores, xlabel,
                            title, log, categorical_x=False):
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    table = pd.DataFrame({
        xlabel: x_values,
        "train_macro_f1_mean": train_mean,
        "train_macro_f1_std": train_std,
        "val_macro_f1_mean": val_mean,
        "val_macro_f1_std": val_std,
    })
    table.to_csv(os.path.join(TABLE_DIR, f"validation_curve_{name}.csv"), index=False)
    log.write(f"\n{title}:")
    log.write(table.to_string(index=False))

    x_plot = range(len(x_values)) if categorical_x else x_values

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_plot, train_mean, "o-", label="Train", color="#4C72B0")
    ax.fill_between(x_plot, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color="#4C72B0")
    ax.plot(x_plot, val_mean, "o-", label="CV Validation", color="#C44E52")
    ax.fill_between(x_plot, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color="#C44E52")
    if categorical_x:
        ax.set_xticks(list(x_plot))
        ax.set_xticklabels(x_values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Macro-F1 score")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"validation_curve_{name}.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Q4: class_weight='balanced' vs unweighted (RF)
# ---------------------------------------------------------------------------
def class_weight_comparison(X_train, y_train, log):
    log.section("Q4: DOES class_weight='balanced' IMPROVE MINORITY-CLASS "
                "DETECTION? (Random Forest, CV)")

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for weight_setting in [None, "balanced"]:
        clf = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            class_weight=weight_setting, n_jobs=-1,
        )
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            clf.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            preds = clf.predict(X_train.iloc[val_idx])
            scores.append(f1_score(y_train.iloc[val_idx], preds, average="macro"))
        results[str(weight_setting)] = scores

    table = pd.DataFrame(results)
    table.loc["mean"] = table.mean()
    table.loc["std"] = table.std()
    table.to_csv(os.path.join(TABLE_DIR, "class_weight_comparison.csv"))
    log.write(table.to_string())
    log.write("\nInterpretation: compare the 'balanced' vs 'None' mean "
              "macro-F1 above. A higher mean for 'balanced' indicates class "
              "weighting meaningfully improves minority-class detection on "
              "this severely imbalanced (534:1) dataset.")


# ---------------------------------------------------------------------------
# Final tuning via RandomizedSearchCV + test set evaluation
# ---------------------------------------------------------------------------
def tune_random_forest(X_train, y_train, log):
    log.section("FINAL RANDOM FOREST TUNING (RandomizedSearchCV)")

    # SPEED TEST: trimmed grid. Restore wider grid later if desired.
    # Removed max_features=None (uses ALL features per split - very slow on
    # 74 columns) and max_depth=None (unconstrained trees) from the search
    # space for this speed-test pass, since combined with n_estimators=200
    # these produce extremely expensive full-depth, full-width trees.
    param_dist = {
        "n_estimators": [100, 150, 200, 300, 400],
        "max_depth": [8, 12, 16, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        # BUG FIX: n_jobs=1 here (was -1). RandomizedSearchCV already
        # parallelizes across candidates x folds via its own n_jobs=-1;
        # also parallelizing each individual RandomForestClassifier caused
        # nested joblib parallelism (processes spawning processes),
        # resulting in severe CPU oversubscription and an apparent "stall".
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        param_distributions=param_dist,
        n_iter=N_RANDOM_SEARCH_ITER,
        scoring=macro_f1_scorer,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    print(f"[RF] Starting RandomizedSearchCV ({N_RANDOM_SEARCH_ITER} candidates x {N_CV_FOLDS} folds)...")
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    log.write(f"RandomForest RandomizedSearchCV completed in {elapsed:.1f}s "
              f"({N_RANDOM_SEARCH_ITER} candidates x {N_CV_FOLDS} folds)")
    log.write(f"Best params: {search.best_params_}")
    log.write(f"Best CV macro-F1: {search.best_score_:.4f}")

    cv_results = pd.DataFrame(search.cv_results_).sort_values(
        "mean_test_score", ascending=False
    )
    cv_results.to_csv(os.path.join(TABLE_DIR, "cv_results_rf.csv"), index=False)

    return search.best_estimator_, search.best_params_, elapsed


def tune_gradient_boosting(X_train, y_train, log):
    log.section("FINAL HIST GRADIENT BOOSTING TUNING (RandomizedSearchCV)")

    # SPEED TEST: trimmed grid. Restore wider grid (more max_iter options,
    # more max_depth/leaf options) once speed is confirmed acceptable.
    param_dist = {
        "max_iter": [100, 150, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [None, 6, 10, 15],
        "max_leaf_nodes": [15, 31, 63, 127],
        "min_samples_leaf": [5, 10, 20],
        "l2_regularization": [0.0, 0.1, 1.0],
    }

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # HistGradientBoostingClassifier supports sample_weight in fit(), so we
    # still address class imbalance via inverse-frequency weighting, exactly
    # as we would have for standard GradientBoostingClassifier.
    class_counts = y_train.value_counts()
    weight_map = (len(y_train) / (len(class_counts) * class_counts)).to_dict()
    sample_weight = y_train.map(weight_map).values
    log.write(f"Using inverse-frequency sample weights for HistGBM: {weight_map}")

    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        param_distributions=param_dist,
        n_iter=N_RANDOM_SEARCH_ITER,
        scoring=macro_f1_scorer,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    print(f"[HistGBM] Starting RandomizedSearchCV ({N_RANDOM_SEARCH_ITER} candidates x {N_CV_FOLDS} folds)...")
    t0 = time.time()
    search.fit(X_train, y_train, sample_weight=sample_weight)
    elapsed = time.time() - t0

    log.write(f"HistGradientBoosting RandomizedSearchCV completed in {elapsed:.1f}s "
              f"({N_RANDOM_SEARCH_ITER} candidates x {N_CV_FOLDS} folds)")
    log.write(f"Best params: {search.best_params_}")
    log.write(f"Best CV macro-F1: {search.best_score_:.4f}")

    cv_results = pd.DataFrame(search.cv_results_).sort_values(
        "mean_test_score", ascending=False
    )
    cv_results.to_csv(os.path.join(TABLE_DIR, "cv_results_gbm.csv"), index=False)

    return search.best_estimator_, search.best_params_, elapsed


def evaluate_on_test_set(model, model_name, X_test, y_test, log):
    log.section(f"TEST SET EVALUATION: {model_name}")

    t0 = time.time()
    preds = model.predict(X_test)
    predict_time = time.time() - t0

    labels_sorted = sorted(y_test.unique())
    target_names = [ATTACK_CAT_NAME_MAP.get(c, str(c)) for c in labels_sorted]

    report_dict = classification_report(
        y_test, preds, labels=labels_sorted, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(os.path.join(TABLE_DIR, f"test_set_classification_report_{model_name}.csv"))
    log.write(report_df.to_string())
    log.write(f"\nPrediction time on test set ({len(X_test)} rows): {predict_time:.3f}s")

    cm = confusion_matrix(y_test, preds, labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix: {model_name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"confusion_matrix_{model_name}.png"), dpi=150)
    plt.close(fig)

    macro_f1 = f1_score(y_test, preds, average="macro")
    weighted_f1 = f1_score(y_test, preds, average="weighted")

    return {
        "model": model_name,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "test_accuracy": report_dict["accuracy"],
        "predict_time_s": predict_time,
    }


def save_feature_importance(model, model_name, feature_names, X_train, y_train, log):
    # NOTE: HistGradientBoostingClassifier does NOT expose
    # feature_importances_ natively (unlike RandomForestClassifier and
    # standard GradientBoostingClassifier). We compute permutation
    # importance instead, which works for any fitted estimator and is
    # arguably a more robust importance measure anyway (it is model-
    # agnostic and accounts for feature interactions/correlation more
    # directly than impurity-based importances). This is worth mentioning
    # explicitly in your report's Implementation section.
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names)
    else:
        from sklearn.inspection import permutation_importance
        log.write(f"\n{model_name}: no native feature_importances_ "
                  f"(expected for HistGradientBoostingClassifier) - "
                  f"computing permutation importance on a subsample instead.")
        # Use a subsample for speed; permutation importance re-scores the
        # model many times and is expensive on 206k rows / full feature set.
        sample_idx = X_train.sample(n=min(20000, len(X_train)), random_state=RANDOM_STATE).index
        result = permutation_importance(
            model, X_train.loc[sample_idx], y_train.loc[sample_idx],
            n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1,
            scoring=macro_f1_scorer,
        )
        importances = pd.Series(result.importances_mean, index=feature_names)

    importances = importances.sort_values(ascending=False)
    importances.to_csv(os.path.join(TABLE_DIR, f"feature_importance_{model_name}.csv"))

    top20 = importances.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    top20.iloc[::-1].plot(kind="barh", ax=ax, color="#55A868")
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top 20 feature importances: {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"feature_importance_{model_name}.png"), dpi=150)
    plt.close(fig)

    log.write(f"\nTop 10 features ({model_name}):")
    log.write(top20.head(10).to_string())


def main():
    ensure_dirs()
    log = Logger(LOG_PATH)

    overall_start = time.time()

    X_train, X_test, y_train, y_test = load_processed_data()
    log.section("DATA LOADED")
    log.write(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    log.write(f"Train class distribution:\n{y_train.value_counts(normalize=True).sort_index()}")
    log.write(f"\n*** SPEED TEST CONFIGURATION: N_CV_FOLDS={N_CV_FOLDS}, "
              f"N_RANDOM_SEARCH_ITER={N_RANDOM_SEARCH_ITER} ***")

    # --- Empirical process: answer tuning questions first ---
    rf_validation_curves(X_train, y_train, log)
    gbm_validation_curves(X_train, y_train, log)
    class_weight_comparison(X_train, y_train, log)

    # --- Final tuned models ---
    best_rf, best_rf_params, rf_tuning_time = tune_random_forest(X_train, y_train, log)
    best_gbm, best_gbm_params, gbm_tuning_time = tune_gradient_boosting(X_train, y_train, log)

    joblib.dump(best_rf, os.path.join(MODEL_DIR, "best_rf.joblib"))
    joblib.dump(best_gbm, os.path.join(MODEL_DIR, "best_gbm.joblib"))

    with open(os.path.join(TABLE_DIR, "best_hyperparameters.json"), "w") as f:
        json.dump({
            "random_forest": best_rf_params,
            "hist_gradient_boosting": best_gbm_params,
        }, f, indent=2, default=str)

    # --- Test set evaluation ---
    rf_results = evaluate_on_test_set(best_rf, "rf", X_test, y_test, log)
    gbm_results = evaluate_on_test_set(best_gbm, "gbm", X_test, y_test, log)

    rf_results["tuning_time_s"] = rf_tuning_time
    gbm_results["tuning_time_s"] = gbm_tuning_time

    # --- Feature importances ---
    log.section("FEATURE IMPORTANCE")
    save_feature_importance(best_rf, "rf", X_train.columns, X_train, y_train, log)
    save_feature_importance(best_gbm, "gbm", X_train.columns, X_train, y_train, log)

    # --- Final comparison table (plug in Assignment 2 numbers manually) ---
    comparison = pd.DataFrame([rf_results, gbm_results])
    comparison.to_csv(os.path.join(TABLE_DIR, "model_comparison_summary.csv"), index=False)

    log.section("FINAL MODEL COMPARISON (RF vs HistGBM) - "
                "add your Assignment 2 model rows manually for the report")
    log.write(comparison.to_string(index=False))

    total_elapsed = time.time() - overall_start
    log.section("TRAINING COMPLETE")
    log.write(f"Total wall-clock time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    log.write(f"Models saved to: {MODEL_DIR}")
    log.write(f"Figures saved to: {FIG_DIR}")
    log.write(f"Tables saved to: {TABLE_DIR}")


if __name__ == "__main__":
    main()