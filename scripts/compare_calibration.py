"""
Compare baseline (no calibration) vs calibration-aware inference,
using the 4-feature model (EDA+TEMP) that is our current best.

Both conditions:
  - Same model: RBF SVM
  - Same LOSO splits
  - Same class weighting

Only difference: per-subject calibration is applied to train and
test features in the calibrated condition.
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
from src.features.eda_features import extract_eda_features
from src.features.temp_features import extract_temp_features
from src.evaluation.loso import run_loso
from src.evaluation.loso_calibrated import run_loso_calibrated


SUBJECTS = [
    "S2","S3","S4","S5","S6","S7","S8","S9","S10","S11",
    "S13","S14","S15","S16","S17",
]

MODEL_NAME = "svm_rbf"

EDA_KEEP = ["mean", "std", "mean_absolute_change"]


def build_feature_dataset():
    """4-feature EDA+TEMP dataset. Same as previous experiments."""
    all_X, all_y, all_g = [], [], []
    for sid in SUBJECTS:
        data = load_subject(sid)
        eda_w, temp_w, _, y = create_all_windows(data)

        rows = []
        for e, t in zip(eda_w, temp_w):
            ef = extract_eda_features(e, sampling_rate=4)
            tf = extract_temp_features(t, sampling_rate=4)
            rows.append(
                [ef[k] for k in EDA_KEEP] + [tf["temp_mean"]]
            )

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
    print("CALIBRATION-AWARE INFERENCE — 4-feature EDA+TEMP model")
    print("=" * 78)

    X, y, groups = build_feature_dataset()
    print(f"\nDataset: X={X.shape}, y={y.shape}, "
          f"subjects={len(np.unique(groups))}")

    # ---------- Condition 1: baseline (no calibration) ----------
    print("\n" + "=" * 78)
    print("CONDITION 1: baseline LOSO (no calibration)")
    print("=" * 78)
    res_base, yt_base, yp_base = run_loso(
        X, y, groups, model_name=MODEL_NAME, verbose=True,
    )
    s_base = summarize(res_base, yt_base, yp_base, "no_calibration")

    # ---------- Condition 2: calibration-aware ----------
    print("\n" + "=" * 78)
    print("CONDITION 2: calibration-aware LOSO")
    print("=" * 78)
    res_cal, yt_cal, yp_cal = run_loso_calibrated(
        X, y, groups, model_name=MODEL_NAME, verbose=True,
    )
    s_cal = summarize(res_cal, yt_cal, yp_cal, "calibrated")

    # ---------- Comparison table ----------
    print("\n" + "=" * 78)
    print("COMPARISON")
    print("=" * 78)
    header = (
        f"{'Condition':<18}"
        f"{'F1(subj mean)':<18}"
        f"{'F1(subj med)':<16}"
        f"{'F1(agg)':<12}"
        f"{'Acc(agg)':<10}"
    )
    print(header)
    print("-" * len(header))

    for s in [s_base, s_cal]:
        print(
            f"{s['label']:<18}"
            f"{s['f1_subj_mean']:<7.4f}±{s['f1_subj_std']:<9.4f}"
            f"{s['f1_subj_median']:<16.4f}"
            f"{s['f1_agg']:<12.4f}"
            f"{s['acc_agg']:<10.4f}"
        )

    # ---------- Paired test ----------
    print("\n" + "=" * 78)
    print("PAIRED TEST (Wilcoxon, per-subject F1)")
    print("=" * 78)

    a = s_base["per_subj_f1"]
    b = s_cal["per_subj_f1"]
    diff = b - a

    print(f"\nCalibrated − Baseline:")
    print(f"  improved: {int(np.sum(diff > 0))}/{len(diff)}")
    print(f"  worse:    {int(np.sum(diff < 0))}/{len(diff)}")
    print(f"  unchanged: {int(np.sum(diff == 0))}/{len(diff)}")
    print(f"  mean Δ:   {diff.mean():+.4f}")
    print(f"  median Δ: {np.median(diff):+.4f}")

    try:
        stat, p = wilcoxon(b, a, zero_method="wilcox", alternative="two-sided")
        print(f"  Wilcoxon p = {p:.4f} "
              f"{'(significant)' if p < 0.05 else '(not significant)'}")
    except ValueError as e:
        print(f"  Wilcoxon failed: {e}")

    # ---------- Per-subject delta table ----------
    print("\n" + "=" * 78)
    print("PER-SUBJECT ΔF1 (calibrated − baseline)")
    print("=" * 78)
    print(f"{'Subject':<10}{'Baseline':<14}{'Calibrated':<14}{'Δ':<10}")
    for i, sid in enumerate(s_base["subjects"]):
        b_i = s_base["per_subj_f1"][i]
        c_i = s_cal["per_subj_f1"][i]
        print(f"{sid:<10}{b_i:<14.4f}{c_i:<14.4f}{c_i - b_i:+.4f}")

    # ---------- Focus on S14 ----------
    print("\n" + "=" * 78)
    print("S14 in detail")
    print("=" * 78)
    s14_base = s_base["per_subj_f1"][s_base["subjects"].index("S14")]
    s14_cal = s_cal["per_subj_f1"][s_cal["subjects"].index("S14")]
    print(f"  Baseline F1:    {s14_base:.4f}")
    print(f"  Calibrated F1:  {s14_cal:.4f}")
    print(f"  Δ:              {s14_cal - s14_base:+.4f}")


if __name__ == "__main__":
    main()