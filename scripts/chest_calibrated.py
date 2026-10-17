"""
Chest multi-class with per-subject calibration.

For each LOSO fold:
  1. For each training subject, fit calibration on their baseline
     windows, then transform their features.
  2. For the test subject, fit calibration on their own baseline
     windows, then transform their features.
  3. Train on calibrated training features, predict on calibrated
     test features.

Calibration = floored std:
    X* = (X - B_s) / max(sigma_s, 0.01)

Same as the wrist pipeline (loso_calibrated_std_floored), but
applied to the 9 chest features.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.preprocessing.wesad_preprocessing import load_subject, create_chest_windows
from src.features.build_chest_features import build_chest_feature_matrix

SUBJECTS = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11",
            "S13","S14","S15","S16","S17"]
CLASS_NAMES = ["baseline", "stress", "amusement", "meditation"]
FLOOR = 0.01


def build_dataset():
    X_by_subj = {}
    y_by_subj = {}
    for sid in SUBJECTS:
        data = load_subject(sid)
        ecg_w, resp_w, y = create_chest_windows(data)
        if len(y) == 0:
            continue
        X, _ = build_chest_feature_matrix(ecg_w, resp_w)
        valid = ~np.isnan(X).any(axis=1)
        X, y = X[valid], y[valid]
        X_by_subj[sid] = X
        y_by_subj[sid] = y
    return X_by_subj, y_by_subj


def calibrate_subject(X, y, floor=FLOOR):
    """Fit on baseline windows, transform all windows."""
    baseline_mask = (y == 0)
    if baseline_mask.sum() < 2:
        return X.copy(), False
    B_s = X[baseline_mask].mean(axis=0)
    sigma_s = X[baseline_mask].std(axis=0)
    sigma_s = np.maximum(sigma_s, floor)
    return (X - B_s) / sigma_s, True


def run_loso(X_by_subj, y_by_subj, calibrate):
    subjects = sorted(X_by_subj.keys())
    all_y_true, all_y_pred = [], []

    for test_subj in subjects:
        train_parts_X, train_parts_y = [], []
        for train_subj in subjects:
            if train_subj == test_subj:
                continue
            Xs, ys = X_by_subj[train_subj], y_by_subj[train_subj]
            if calibrate:
                Xs, _ = calibrate_subject(Xs, ys)
            train_parts_X.append(Xs)
            train_parts_y.append(ys)

        X_train = np.vstack(train_parts_X)
        y_train = np.concatenate(train_parts_y)

        X_test = X_by_subj[test_subj]
        y_test = y_by_subj[test_subj]
        if calibrate:
            X_test, _ = calibrate_subject(X_test, y_test)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced",
                decision_function_shape="ovr",
                random_state=42,
            )),
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    macro = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    per_class = f1_score(all_y_true, all_y_pred, labels=[0, 1, 2, 3],
                         average=None, zero_division=0)
    return all_y_true, all_y_pred, macro, per_class


def main():
    print("=" * 70)
    print("CHEST MULTI-CLASS — WITH vs WITHOUT CALIBRATION")
    print("=" * 70)

    X_by_subj, y_by_subj = build_dataset()
    print(f"\nSubjects: {len(X_by_subj)}")
    total = sum(len(y) for y in y_by_subj.values())
    print(f"Total windows: {total}")

    # Condition 1: no calibration
    print("\n" + "-" * 70)
    print("CONDITION: no calibration")
    print("-" * 70)
    yt1, yp1, macro1, per_class1 = run_loso(X_by_subj, y_by_subj,
                                            calibrate=False)
    print(classification_report(yt1, yp1, labels=[0, 1, 2, 3],
                                target_names=CLASS_NAMES,
                                zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(yt1, yp1, labels=[0, 1, 2, 3]))

    # Condition 2: with calibration
    print("\n" + "-" * 70)
    print("CONDITION: with calibration")
    print("-" * 70)
    yt2, yp2, macro2, per_class2 = run_loso(X_by_subj, y_by_subj,
                                            calibrate=True)
    print(classification_report(yt2, yp2, labels=[0, 1, 2, 3],
                                target_names=CLASS_NAMES,
                                zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(yt2, yp2, labels=[0, 1, 2, 3]))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<22}{'Macro F1':<12}"
          f"{'Baseline':<10}{'Stress':<10}{'Amuse':<10}{'Meditation':<10}")
    print("-" * 74)
    print(f"{'no calibration':<22}{macro1:<12.4f}"
          f"{per_class1[0]:<10.4f}{per_class1[1]:<10.4f}"
          f"{per_class1[2]:<10.4f}{per_class1[3]:<10.4f}")
    print(f"{'with calibration':<22}{macro2:<12.4f}"
          f"{per_class2[0]:<10.4f}{per_class2[1]:<10.4f}"
          f"{per_class2[2]:<10.4f}{per_class2[3]:<10.4f}")
    print(f"{'Δ':<22}{macro2-macro1:<+12.4f}"
          f"{per_class2[0]-per_class1[0]:<+10.4f}"
          f"{per_class2[1]-per_class1[1]:<+10.4f}"
          f"{per_class2[2]-per_class1[2]:<+10.4f}"
          f"{per_class2[3]-per_class1[3]:<+10.4f}")


if __name__ == "__main__":
    main()