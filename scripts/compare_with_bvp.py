"""
Compare EDA+TEMP (4 features) vs EDA+TEMP+BVP (7 features).

Model: RBF SVM.
Evaluation: LOSO, 15 folds.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, accuracy_score

from src.preprocessing.wesad_preprocessing import (
    load_subject,
    create_all_windows,
)
from src.features.build_all_features import (
    build_eda_temp_matrix,
    build_eda_temp_bvp_matrix,
)
from src.evaluation.loso import run_loso


SUBJECTS = [
    "S2","S3","S4","S5","S6","S7","S8","S9","S10","S11",
    "S13","S14","S15","S16","S17",
]

MODEL_NAME = "svm_rbf"


def build_dataset(include_bvp):
    """Build combined dataset. Returns X, y, groups."""
    all_X, all_y, all_g = [], [], []

    for sid in SUBJECTS:
        data = load_subject(sid)
        eda_w, temp_w, bvp_w, y = create_all_windows(data)

        if include_bvp:
            X, _, kept = build_eda_temp_bvp_matrix(eda_w, temp_w, bvp_w)
            y = y[kept]
        else:
            X, _ = build_eda_temp_matrix(eda_w, temp_w)

        all_X.append(X)
        all_y.append(y)
        all_g.append(np.full(len(y), sid, dtype=object))

    return np.vstack(all_X), np.concatenate(all_y), np.concatenate(all_g)


def run_one(name, X, y, groups):
    print(f"\n{'-' * 78}")
    print(f"FEATURE SET: {name}  (X.shape = {X.shape})")
    print(f"{'-' * 78}")

    results, y_true, y_pred = run_loso(
        X, y, groups, model_name=MODEL_NAME, verbose=False,
    )
    f1s = np.array([r["f1"] for r in results])

    summary = {
        "name": name,
        "n_features": X.shape[1],
        "f1_subj_mean": f1s.mean(),
        "f1_subj_std":  f1s.std(),
        "f1_agg":       f1_score(y_true, y_pred, zero_division=0),
        "acc_agg":      accuracy_score(y_true, y_pred),
        "per_subj_f1":  f1s,
        "subjects":     [r["subject"] for r in results],
    }

    print(f"  F1(subj) = {summary['f1_subj_mean']:.4f} ± "
          f"{summary['f1_subj_std']:.4f}")
    print(f"  F1(agg)  = {summary['f1_agg']:.4f}")
    print(f"  Acc(agg) = {summary['acc_agg']:.4f}")

    return summary


def main():
    print("=" * 78)
    print("EDA+TEMP vs EDA+TEMP+BVP — RBF SVM, LOSO, 15 subjects")
    print("=" * 78)

    X_base, y_base, g_base = build_dataset(include_bvp=False)
    X_bvp,  y_bvp,  g_bvp  = build_dataset(include_bvp=True)

    print(f"\nDataset sizes:")
    print(f"  EDA+TEMP:      X={X_base.shape}, y={y_base.shape}")
    print(f"  EDA+TEMP+BVP:  X={X_bvp.shape},  y={y_bvp.shape}")

    s_base = run_one("EDA+TEMP (4)",     X_base, y_base, g_base)
    s_bvp  = run_one("EDA+TEMP+BVP (7)", X_bvp,  y_bvp,  g_bvp)

    # Paired test — only makes sense if subjects align
    print("\n" + "=" * 78)
    print("PAIRED TEST (Wilcoxon, per-subject F1)")
    print("=" * 78)

    base_f1 = dict(zip(s_base["subjects"], s_base["per_subj_f1"]))
    bvp_f1  = dict(zip(s_bvp["subjects"],  s_bvp["per_subj_f1"]))

    common = sorted(set(base_f1) & set(bvp_f1))
    a = np.array([base_f1[s] for s in common])
    b = np.array([bvp_f1[s]  for s in common])
    diff = b - a

    print(f"\nSubjects compared: {len(common)}")
    print(f"  improved: {int(np.sum(diff > 0))}/{len(common)}")
    print(f"  worse:    {int(np.sum(diff < 0))}/{len(common)}")
    print(f"  mean Δ:   {diff.mean():+.4f}")

    if len(common) >= 6:
        try:
            stat, p = wilcoxon(b, a, zero_method="wilcox",
                               alternative="two-sided")
            print(f"  p = {p:.4f} "
                  f"{'(significant)' if p < 0.05 else '(not significant)'}")
        except ValueError as e:
            print(f"  Wilcoxon failed: {e}")

    # Per-subject delta table
    print("\nPer-subject ΔF1 (BVP − no BVP):")
    print("-" * 50)
    print(f"{'Subject':<10}{'EDA+TEMP':<14}{'EDA+TEMP+BVP':<16}{'Δ':<10}")
    for s in common:
        print(f"{s:<10}{base_f1[s]:<14.4f}"
              f"{bvp_f1[s]:<16.4f}"
              f"{bvp_f1[s] - base_f1[s]:+.4f}")


if __name__ == "__main__":
    main()