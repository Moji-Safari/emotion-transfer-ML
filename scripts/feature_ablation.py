"""
Feature ablation for the WESAD stress-recognition project.

Strategy
--------
Baseline: all 9 EDA features, RBF SVM, LOSO.
Then for each feature f in {9 features}:
    Drop f, keep the other 8, run the same LOSO with the same model,
    record F1.

Interpretation
--------------
delta_f = F1(baseline) - F1(drop f)

    delta_f > 0   : f carried unique information; dropping it hurt.
    delta_f ~ 0   : f was redundant (another feature compensated) or useless.
    delta_f < 0   : f was noise; the model did better without it.

The baseline result should match the RBF SVM row from
scripts/compare_models.py.

Reference
---------
Schmidt et al. (2018), "Introducing WESAD" — ICMI 2018
https://doi.org/10.1145/3242969.3242985

Guyon & Elisseeff (2003), "An Introduction to Variable and
Feature Selection" — JMLR.
https://jmlr.org/papers/v3/guyon03a.html
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from src.data.build_dataset import build_all_subjects_dataset
from src.evaluation.loso import run_loso


# -------------------------------------------------------------------
# The feature order MUST match the dict insertion order in
# src/features/eda_features.py:extract_eda_features().
# If you ever change that file, update this list.
# -------------------------------------------------------------------
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

# Which model to use. RBF SVM won the model comparison.
MODEL_NAME = "svm_rbf"


def summarize_fold_results(results, y_true, y_pred):
    f1s = np.array([r["f1"] for r in results])
    return {
        "f1_subj_mean": f1s.mean(),
        "f1_subj_std":  f1s.std(),
        "f1_agg":       f1_score(y_true, y_pred, zero_division=0),
        "acc_agg":      accuracy_score(y_true, y_pred),
    }


def run_with_features(X, y, groups, keep_indices, label):
    """Run LOSO on X restricted to keep_indices. Return summary."""
    X_sub = X[:, keep_indices]

    print(f"\n{'-' * 78}")
    print(f"RUN: {label}")
    print(f"  features kept: {[FEATURE_NAMES[i] for i in keep_indices]}")
    print(f"  shape: {X_sub.shape}")
    print(f"{'-' * 78}")

    results, y_true, y_pred = run_loso(
        X_sub, y, groups,
        model_name=MODEL_NAME,
        verbose=False,  # keep output short
    )
    return summarize_fold_results(results, y_true, y_pred)


def main():
    print("=" * 78)
    print("FEATURE ABLATION — RBF SVM, LOSO, 15 subjects")
    print("=" * 78)

    X, y, groups = build_all_subjects_dataset()

    print(f"\nDataset: X={X.shape}, y={y.shape}, "
          f"subjects={len(np.unique(groups))}")

    if X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Feature count mismatch: X has {X.shape[1]} columns, "
            f"but FEATURE_NAMES has {len(FEATURE_NAMES)}. "
            f"Update FEATURE_NAMES to match the actual feature order."
        )

    # ---------- Baseline: all features ----------
    all_indices = list(range(len(FEATURE_NAMES)))
    baseline = run_with_features(
        X, y, groups, all_indices, label="BASELINE (all 9 features)"
    )

    print(f"\nBASELINE: F1(subj) = "
          f"{baseline['f1_subj_mean']:.4f} ± "
          f"{baseline['f1_subj_std']:.4f}")

    # ---------- Leave-one-feature-out ----------
    ablation_results = {}

    for drop_idx, drop_name in enumerate(FEATURE_NAMES):
        keep = [i for i in all_indices if i != drop_idx]
        summary = run_with_features(
            X, y, groups, keep, label=f"DROP '{drop_name}'"
        )
        ablation_results[drop_name] = summary

        delta = baseline["f1_subj_mean"] - summary["f1_subj_mean"]
        sign = "+" if delta >= 0 else ""
        print(f"  → F1(subj) = {summary['f1_subj_mean']:.4f}  "
              f"(Δ = {sign}{delta:.4f})")

    # ---------- Final table ----------
    print("\n" + "=" * 78)
    print("FEATURE ABLATION — FINAL TABLE")
    print("=" * 78)

    header = (
        f"{'Dropped feature':<24}"
        f"{'F1(subj)':<18}"
        f"{'ΔF1':<10}"
        f"{'F1(agg)':<10}"
    )
    print(header)
    print("-" * len(header))

    # Sort by delta descending: biggest loss first (= most important)
    rows = []
    for name, s in ablation_results.items():
        delta = baseline["f1_subj_mean"] - s["f1_subj_mean"]
        rows.append((name, s, delta))
    rows.sort(key=lambda r: r[2], reverse=True)

    for name, s, delta in rows:
        sign = "+" if delta >= 0 else ""
        print(
            f"{name:<24}"
            f"{s['f1_subj_mean']:<7.4f}±"
            f"{s['f1_subj_std']:<8.4f}"
            f"{sign}{delta:<9.4f}"
            f"{s['f1_agg']:<10.4f}"
        )

    print("-" * len(header))
    print(
        f"{'BASELINE (all 9)':<24}"
        f"{baseline['f1_subj_mean']:<7.4f}±"
        f"{baseline['f1_subj_std']:<8.4f}"
        f"{'--':<10}"
        f"{baseline['f1_agg']:<10.4f}"
    )

    # ---------- Interpretation hints ----------
    print("\n" + "=" * 78)
    print("HOW TO READ THIS TABLE")
    print("=" * 78)
    print("ΔF1 > 0  : dropping this feature HURT. Feature carries unique info.")
    print("ΔF1 ≈ 0  : dropping this feature made NO difference. Redundant.")
    print("ΔF1 < 0  : dropping this feature HELPED. Feature was noise.")
    print()
    print("Caution: features are correlated. Removing 'median' may cost")
    print("nothing because 'mean' compensates. That does NOT mean median")
    print("is useless — it means median is redundant GIVEN mean.")


if __name__ == "__main__":
    main()