"""
Compare EDA-only, TEMP-only, and EDA+TEMP.

Model: RBF SVM (same as previous experiments).
Evaluation: LOSO, 15 folds.
Feature sets:
    EDA-only   : 3 features (mean, std, mean_absolute_change)
    TEMP-only  : 3 features (mean, std, slope)
    EDA+TEMP   : 6 features
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, accuracy_score

from src.preprocessing.wesad_preprocessing import (
    load_subject,
    create_eda_temp_windows,
)
from src.features.build_combined_features import (
    build_eda_only_matrix,
    build_temp_only_matrix,
    build_combined_feature_matrix,
)
from src.evaluation.loso import run_loso


SUBJECTS = [
    "S2","S3","S4","S5","S6","S7","S8","S9","S10","S11",
    "S13","S14","S15","S16","S17",
]

MODEL_NAME = "svm_rbf"


def build_dataset(kind):
    """
    kind in {'eda', 'temp', 'combined'}.
    Returns X, y, groups.
    """
    all_X, all_y, all_groups = [], [], []

    for sid in SUBJECTS:
        data = load_subject(sid)
        eda_w, temp_w, y = create_eda_temp_windows(data)

        if kind == "eda":
            X, _ = build_eda_only_matrix(eda_w)
        elif kind == "temp":
            X, _ = build_temp_only_matrix(temp_w)
        elif kind == "combined":
            X, _ = build_combined_feature_matrix(eda_w, temp_w)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        all_X.append(X)
        all_y.append(y)
        all_groups.append(np.full(len(y), sid, dtype=object))

    return np.vstack(all_X), np.concatenate(all_y), np.concatenate(all_groups)


def run_one(kind, X, y, groups):
    print(f"\n{'-' * 78}")
    print(f"FEATURES: {kind}  (X.shape = {X.shape})")
    print(f"{'-' * 78}")

    results, y_true, y_pred = run_loso(
        X, y, groups, model_name=MODEL_NAME, verbose=False,
    )

    f1s = np.array([r["f1"] for r in results])
    summary = {
        "kind": kind,
        "n_features": X.shape[1],
        "f1_subj_mean": f1s.mean(),
        "f1_subj_std":  f1s.std(),
        "f1_agg":       f1_score(y_true, y_pred, zero_division=0),
        "acc_agg":      accuracy_score(y_true, y_pred),
        "per_subj_f1":  f1s,
    }

    print(f"  F1(subj) = {summary['f1_subj_mean']:.4f} ± "
          f"{summary['f1_subj_std']:.4f}")
    print(f"  F1(agg)  = {summary['f1_agg']:.4f}")
    print(f"  Acc(agg) = {summary['acc_agg']:.4f}")

    return summary


def main():
    print("=" * 78)
    print("EDA vs TEMP vs EDA+TEMP — RBF SVM, LOSO, 15 subjects")
    print("=" * 78)

    summaries = {}
    for kind in ["eda", "temp", "combined"]:
        X, y, groups = build_dataset(kind)
        summaries[kind] = run_one(kind, X, y, groups)

    # ---------- Paired tests ----------
    print("\n" + "=" * 78)
    print("PAIRED TESTS (Wilcoxon signed-rank, per-subject F1)")
    print("=" * 78)

    eda_f1  = summaries["eda"]["per_subj_f1"]
    temp_f1 = summaries["temp"]["per_subj_f1"]
    comb_f1 = summaries["combined"]["per_subj_f1"]

    def report_pair(a, b, name_a, name_b):
        diff = b - a
        try:
            stat, p = wilcoxon(b, a, zero_method="wilcox",
                               alternative="two-sided")
        except ValueError:
            stat, p = float("nan"), float("nan")
        print(f"\n{name_b} − {name_a}:")
        print(f"  improved: {int(np.sum(diff > 0)):>2}/15")
        print(f"  worse:    {int(np.sum(diff < 0)):>2}/15")
        print(f"  mean diff: {diff.mean():+.4f}")
        print(f"  Wilcoxon p = {p:.4f} "
              f"{'(significant)' if p < 0.05 else '(not significant)'}")

    report_pair(eda_f1, temp_f1, "EDA-only", "TEMP-only")
    report_pair(eda_f1, comb_f1, "EDA-only", "EDA+TEMP")
    report_pair(temp_f1, comb_f1, "TEMP-only", "EDA+TEMP")

    # ---------- Final table ----------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = (
        f"{'Features':<14}"
        f"{'n_feat':<8}"
        f"{'F1(subj)':<18}"
        f"{'F1(agg)':<10}"
        f"{'Acc(agg)':<10}"
    )
    print(header)
    print("-" * len(header))

    for kind in ["eda", "temp", "combined"]:
        s = summaries[kind]
        print(
            f"{kind:<14}"
            f"{s['n_features']:<8}"
            f"{s['f1_subj_mean']:<7.4f}±"
            f"{s['f1_subj_std']:<8.4f}"
            f"{s['f1_agg']:<10.4f}"
            f"{s['acc_agg']:<10.4f}"
        )


if __name__ == "__main__":
    main()