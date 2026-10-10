"""
Compare three conditions:
  1. no_calibration        — raw features
  2. calibration_mean      — X* = (X - B_s) / |B_s|
  3. calibration_std       — X* = (X - B_s) / sigma_s

Model: RBF SVM. Features: 4 (3 EDA + 1 TEMP).
Evaluation: LOSO, 15 folds.

Focus: does std-based calibration fix S16 (which was broken by
mean-based calibration) without losing the gains on the other
subjects?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, accuracy_score

from src.preprocessing.wesad_preprocessing import (
    load_subject, create_all_windows,
)
from src.features.eda_features import extract_eda_features
from src.features.temp_features import extract_temp_features
from src.evaluation.loso import run_loso
from src.evaluation.loso_calibrated import run_loso_calibrated
from src.evaluation.loso_calibrated_std import run_loso_calibrated_std


SUBJECTS = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11",
            "S13","S14","S15","S16","S17"]
MODEL_NAME = "svm_rbf"
EDA_KEEP = ["mean", "std", "mean_absolute_change"]


def build_feature_dataset():
    all_X, all_y, all_g = [], [], []
    for sid in SUBJECTS:
        data = load_subject(sid)
        eda_w, temp_w, _, y = create_all_windows(data)
        rows = []
        for e, t in zip(eda_w, temp_w):
            ef = extract_eda_features(e, sampling_rate=4)
            tf = extract_temp_features(t, sampling_rate=4)
            rows.append([ef[k] for k in EDA_KEEP] + [tf["temp_mean"]])
        all_X.append(np.asarray(rows, dtype=float))
        all_y.append(y)
        all_g.append(np.full(len(y), sid, dtype=object))
    return np.vstack(all_X), np.concatenate(all_y), np.concatenate(all_g)


def summarize(results, y_true, y_pred, label):
    f1s = np.array([r["f1"] for r in results])
    return {
        "label": label,
        "f1_subj_mean":  f1s.mean(),
        "f1_subj_median": float(np.median(f1s)),
        "f1_subj_std":   f1s.std(),
        "f1_agg":        f1_score(y_true, y_pred, zero_division=0),
        "acc_agg":       accuracy_score(y_true, y_pred),
        "per_subj_f1":   f1s,
        "subjects":      [r["subject"] for r in results],
    }


def main():
    print("=" * 78)
    print("CALIBRATION COMPARISON v2")
    print("=" * 78)

    X, y, groups = build_feature_dataset()
    print(f"\nDataset: X={X.shape}, y={y.shape}, "
          f"subjects={len(np.unique(groups))}")

    # Condition 1
    print("\n" + "=" * 78)
    print("CONDITION 1: no calibration")
    print("=" * 78)
    r1, yt1, yp1 = run_loso(X, y, groups,
                            model_name=MODEL_NAME, verbose=False)
    s1 = summarize(r1, yt1, yp1, "no_calibration")

    # Condition 2
    print("\n" + "=" * 78)
    print("CONDITION 2: mean-based calibration")
    print("=" * 78)
    r2, yt2, yp2 = run_loso_calibrated(X, y, groups,
                                       model_name=MODEL_NAME, verbose=False)
    s2 = summarize(r2, yt2, yp2, "calibration_mean")

    # Condition 3
    print("\n" + "=" * 78)
    print("CONDITION 3: std-based calibration")
    print("=" * 78)
    r3, yt3, yp3 = run_loso_calibrated_std(X, y, groups,
                                           model_name=MODEL_NAME, verbose=False)
    s3 = summarize(r3, yt3, yp3, "calibration_std")

    # ---------- Comparison table ----------
    print("\n" + "=" * 78)
    print("COMPARISON")
    print("=" * 78)
    header = (
        f"{'Condition':<20}"
        f"{'F1(mean)':<18}"
        f"{'F1(median)':<14}"
        f"{'F1(agg)':<12}"
        f"{'Acc(agg)':<10}"
    )
    print(header)
    print("-" * len(header))

    for s in [s1, s2, s3]:
        print(
            f"{s['label']:<20}"
            f"{s['f1_subj_mean']:<7.4f}±{s['f1_subj_std']:<9.4f}"
            f"{s['f1_subj_median']:<14.4f}"
            f"{s['f1_agg']:<12.4f}"
            f"{s['acc_agg']:<10.4f}"
        )

    # ---------- Paired tests ----------
    print("\n" + "=" * 78)
    print("PAIRED TESTS (Wilcoxon, per-subject F1)")
    print("=" * 78)

    def pair(a, b, name_a, name_b):
        diff = b - a
        try:
            stat, p = wilcoxon(b, a, zero_method="wilcox",
                               alternative="two-sided")
        except ValueError:
            p = float("nan")
        print(f"\n{name_b} − {name_a}:")
        print(f"  improved: {int(np.sum(diff > 0))}/{len(diff)}")
        print(f"  worse:    {int(np.sum(diff < 0))}/{len(diff)}")
        print(f"  mean Δ:   {diff.mean():+.4f}")
        print(f"  p = {p:.4f} "
              f"{'(significant)' if p < 0.05 else '(not significant)'}")

    pair(s1["per_subj_f1"], s3["per_subj_f1"],
         "no_calibration", "calibration_std")
    pair(s2["per_subj_f1"], s3["per_subj_f1"],
         "calibration_mean", "calibration_std")

    # ---------- Per-subject table ----------
    print("\n" + "=" * 78)
    print("PER-SUBJECT F1 (all three conditions)")
    print("=" * 78)
    print(f"{'Subj':<8}{'None':<10}{'Mean':<10}{'Std':<10}"
          f"{'Δstd−mean':<12}")
    print("-" * 50)
    subs = s1["subjects"]
    for i, sid in enumerate(subs):
        n_ = s1["per_subj_f1"][i]
        m_ = s2["per_subj_f1"][i]
        st = s3["per_subj_f1"][i]
        print(f"{sid:<8}{n_:<10.4f}{m_:<10.4f}{st:<10.4f}"
              f"{st - m_:+12.4f}")

    # ---------- S14 and S16 focus ----------
    print("\n" + "=" * 78)
    print("S14 AND S16 FOCUS")
    print("=" * 78)
    for sid in ["S14", "S16"]:
        i = subs.index(sid)
        print(f"\n{sid}:")
        print(f"  no_calibration:   {s1['per_subj_f1'][i]:.4f}")
        print(f"  calibration_mean: {s2['per_subj_f1'][i]:.4f}")
        print(f"  calibration_std:  {s3['per_subj_f1'][i]:.4f}")


if __name__ == "__main__":
    main()