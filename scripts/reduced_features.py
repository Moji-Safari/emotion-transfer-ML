"""
Reduced-feature experiment + paired test.

Motivation
----------
Feature ablation showed:
  - All 9 features are highly correlated (max ΔF1 = 0.026).
  - Dropping `std` gave the largest improvement (+0.026 F1).
  - The 9 features may reduce to ~3 underlying concepts:
        central level  : mean, median, min, max, range
        dispersion     : std, mean_absolute_change, std_change
        temporal trend : slope

This script tests whether:
  1. A small subset of features matches the 9-feature baseline.
  2. The improvement from dropping `std` is statistically real,
     using a Wilcoxon signed-rank test on per-subject F1 across folds.

References
----------
- Guyon & Elisseeff (2003), JMLR. Feature selection principles.
  https://jmlr.org/papers/v3/guyon03a.html
- Wilcoxon (1945), "Individual Comparisons by Ranking Methods."
  Biometrics Bulletin, 1(6), 80-83.
  (See also: scipy.stats.wilcoxon documentation.)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, accuracy_score

from src.data.build_dataset import build_all_subjects_dataset
from src.evaluation.loso import run_loso


FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "median",
    "mean_absolute_change",
    "std_change",
    "slope",
]

MODEL_NAME = "svm_rbf"


def indices_of(names):
    """Map feature names to their column indices."""
    return [FEATURE_NAMES.index(n) for n in names]


def run_and_report(X, y, groups, keep_indices, label):
    """Run LOSO, return (summary_dict, per_subject_f1_array)."""
    X_sub = X[:, keep_indices]

    print(f"\n{'-' * 78}")
    print(f"RUN: {label}")
    print(f"  n_features = {len(keep_indices)}")
    print(f"  features   = {[FEATURE_NAMES[i] for i in keep_indices]}")
    print(f"{'-' * 78}")

    results, y_true, y_pred = run_loso(
        X_sub, y, groups,
        model_name=MODEL_NAME,
        verbose=False,
    )

    per_subj_f1 = np.array([r["f1"] for r in results])
    summary = {
        "label": label,
        "n_features": len(keep_indices),
        "features": [FEATURE_NAMES[i] for i in keep_indices],
        "f1_subj_mean": per_subj_f1.mean(),
        "f1_subj_std":  per_subj_f1.std(),
        "f1_agg":       f1_score(y_true, y_pred, zero_division=0),
        "acc_agg":      accuracy_score(y_true, y_pred),
        "per_subj_f1":  per_subj_f1,
    }

    print(f"  F1(subj) = {summary['f1_subj_mean']:.4f} ± "
          f"{summary['f1_subj_std']:.4f}")
    return summary


def main():
    print("=" * 78)
    print("REDUCED-FEATURE EXPERIMENT — RBF SVM, LOSO, 15 subjects")
    print("=" * 78)

    X, y, groups = build_all_subjects_dataset()

    print(f"\nDataset: X={X.shape}, y={y.shape}, "
          f"subjects={len(np.unique(groups))}")

    # ---------------- Configurations ----------------
    configs = [
        ("all_9",              FEATURE_NAMES),
        ("drop_std",           [f for f in FEATURE_NAMES if f != "std"]),
        ("3_concepts",         ["mean", "std", "slope"]),
        ("3_dynamics",         ["mean", "std", "mean_absolute_change"]),
        ("2_level_disp",       ["mean", "std"]),
        ("1_mean",             ["mean"]),
    ]

    summaries = {}
    for label, feats in configs:
        idx = indices_of(feats)
        summaries[label] = run_and_report(X, y, groups, idx, label)

    # ---------------- Paired test: all_9 vs drop_std ----------------
    print("\n" + "=" * 78)
    print("PAIRED TEST — all_9 vs drop_std (per-subject F1 across 15 folds)")
    print("=" * 78)

    f1_all = summaries["all_9"]["per_subj_f1"]
    f1_drop = summaries["drop_std"]["per_subj_f1"]
    diff = f1_drop - f1_all  # positive means drop_std is better

    n_improved = int(np.sum(diff > 0))
    n_worse = int(np.sum(diff < 0))
    n_same = int(np.sum(diff == 0))

    print(f"\nPer-subject difference (drop_std − all_9):")
    print(f"  improved (positive): {n_improved}")
    print(f"  worse    (negative): {n_worse}")
    print(f"  identical          : {n_same}")
    print(f"  mean diff          : {diff.mean():+.4f}")
    print(f"  median diff        : {np.median(diff):+.4f}")

    # Wilcoxon signed-rank test (two-sided)
    try:
        stat, p = wilcoxon(f1_drop, f1_all, zero_method="wilcox",
                           alternative="two-sided")
        print(f"\n  Wilcoxon signed-rank: statistic = {stat}, "
              f"p-value = {p:.4f}")
        if p < 0.05:
            print("  => Statistically significant at α = 0.05.")
        else:
            print("  => NOT statistically significant at α = 0.05.")
            print("     The apparent +0.026 improvement may be noise.")
    except ValueError as e:
        print(f"\n  Wilcoxon test failed: {e}")
        print("  (This happens if all differences are zero.)")

    # ---------------- Final table ----------------
    print("\n" + "=" * 78)
    print("REDUCED-FEATURE SUMMARY")
    print("=" * 78)

    header = (
        f"{'Config':<18}"
        f"{'n_feat':<8}"
        f"{'F1(subj)':<18}"
        f"{'ΔF1 vs all_9':<15}"
        f"{'F1(agg)':<10}"
    )
    print(header)
    print("-" * len(header))

    baseline = summaries["all_9"]["f1_subj_mean"]

    # Sort by per-subject F1 descending
    ordered = sorted(
        summaries.items(),
        key=lambda kv: kv[1]["f1_subj_mean"],
        reverse=True,
    )

    for label, s in ordered:
        delta = s["f1_subj_mean"] - baseline
        sign = "+" if delta >= 0 else ""
        print(
            f"{label:<18}"
            f"{s['n_features']:<8}"
            f"{s['f1_subj_mean']:<7.4f}±"
            f"{s['f1_subj_std']:<8.4f}"
            f"{sign}{delta:<13.4f}"
            f"{s['f1_agg']:<10.4f}"
        )

    # ---------------- Interpretation hints ----------------
    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("- If a 3-feature config matches all_9 within ±0.02 F1,")
    print("  the other 6 features are redundant. Use the 3.")
    print("- If 1_mean already reaches close to all_9's F1, then even")
    print("  dispersion and trend are not adding much.")
    print("- The Wilcoxon p-value tells you whether the std improvement")
    print("  is real. p < 0.05 = real. Otherwise treat it as noise.")


if __name__ == "__main__":
    main()