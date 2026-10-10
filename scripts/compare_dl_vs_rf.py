"""
Compare the calibrated CNN-LSTM (raw EDA) against the calibrated
RBF SVM + handcrafted features (the current best model).

Both use:
    - Same 15 subjects
    - Same LOSO splits
    - Same per-subject calibration
    - Same binary labels

The only difference is representation and model:
    - RBF SVM: 4 handcrafted features per window
    - CNN-LSTM: raw calibrated EDA waveform (120 samples)
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
from src.evaluation.loso_calibrated_std_floor import (
    run_loso_calibrated_std_floored,
)
from src.evaluation.loso_deep import run_loso_deep


SUBJECTS = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11",
            "S13","S14","S15","S16","S17"]
EDA_KEEP = ["mean", "std", "mean_absolute_change"]
FLOOR = 0.01


def build_feature_dataset():
    """4-feature EDA+TEMP dataset (for the SVM)."""
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


def build_raw_dataset():
    """Raw EDA windows (for the CNN-LSTM), organised per subject."""
    eda_by_subj = {}
    y_by_subj = {}
    for sid in SUBJECTS:
        data = load_subject(sid)
        eda_w, _, _, y = create_all_windows(data)
        eda_by_subj[sid] = eda_w.astype(np.float32)
        y_by_subj[sid] = y.astype(np.int32)
    return eda_by_subj, y_by_subj


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
    print("CNN-LSTM (raw EDA) vs RBF SVM (handcrafted features)")
    print("=" * 78)

    # ---------- SVM baseline (handcrafted features + floor cal) ----------
    print("\nBuilding feature dataset for SVM...")
    X_feat, y_feat, g_feat = build_feature_dataset()
    print(f"  X={X_feat.shape}, y={y_feat.shape}")

    print("\nRunning SVM + floored std calibration (this is the current best)...")
    res_svm, yt_svm, yp_svm = run_loso_calibrated_std_floored(
        X_feat, y_feat, g_feat,
        model_name="svm_rbf",
        floor=FLOOR,
        verbose=True,
    )
    s_svm = summarize(res_svm, yt_svm, yp_svm, "svm_calibrated_std_floor")

    # ---------- CNN-LSTM on raw EDA ----------
    print("\nBuilding raw EDA dataset for CNN-LSTM...")
    eda_by_subj, y_by_subj = build_raw_dataset()
    print(f"  subjects: {len(eda_by_subj)}")

    print("\nRunning CNN-LSTM on calibrated raw EDA...")
    print("(this may take 5-15 minutes depending on your CPU/GPU)")
    res_dl, yt_dl, yp_dl = run_loso_deep(
        eda_by_subj, y_by_subj, SUBJECTS,
        epochs=100, patience=10, verbose=0,
    )
    s_dl = summarize(res_dl, yt_dl, yp_dl, "cnn_lstm_raw_eda")

    # ---------- Comparison ----------
    print("\n" + "=" * 78)
    print("COMPARISON")
    print("=" * 78)
    header = (
        f"{'Model':<26}"
        f"{'F1(mean)':<18}"
        f"{'F1(med)':<12}"
        f"{'F1(agg)':<12}"
        f"{'Acc(agg)':<10}"
    )
    print(header)
    print("-" * len(header))

    for s in [s_svm, s_dl]:
        print(
            f"{s['label']:<26}"
            f"{s['f1_subj_mean']:<7.4f}±{s['f1_subj_std']:<9.4f}"
            f"{s['f1_subj_median']:<12.4f}"
            f"{s['f1_agg']:<12.4f}"
            f"{s['acc_agg']:<10.4f}"
        )

    # ---------- Paired test ----------
    print("\n" + "=" * 78)
    print("PAIRED TEST (Wilcoxon, per-subject F1)")
    print("=" * 78)
    a = s_svm["per_subj_f1"]
    b = s_dl["per_subj_f1"]
    diff = b - a
    print(f"\nCNN-LSTM − SVM:")
    print(f"  improved: {int(np.sum(diff > 0))}/{len(diff)}")
    print(f"  worse:    {int(np.sum(diff < 0))}/{len(diff)}")
    print(f"  mean Δ:   {diff.mean():+.4f}")
    try:
        stat, p = wilcoxon(b, a, zero_method="wilcox", alternative="two-sided")
        print(f"  p = {p:.4f} "
              f"{'(significant)' if p < 0.05 else '(not significant)'}")
    except ValueError as e:
        print(f"  Wilcoxon failed: {e}")

    # ---------- Per-subject ----------
    print("\n" + "=" * 78)
    print("PER-SUBJECT F1")
    print("=" * 78)
    print(f"{'Subj':<8}{'SVM':<12}{'CNN-LSTM':<12}{'Δ':<10}")
    for i, sid in enumerate(s_svm["subjects"]):
        svm_f1 = s_svm["per_subj_f1"][i]
        dl_f1  = s_dl["per_subj_f1"][i]
        print(f"{sid:<8}{svm_f1:<12.4f}{dl_f1:<12.4f}{dl_f1 - svm_f1:+.4f}")


if __name__ == "__main__":
    main()