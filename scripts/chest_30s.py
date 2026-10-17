"""
Chest multi-class with 30-second windows.

Drops hrv_lfhf because the LF/HF ratio cannot be reliably
computed on 30s windows (LF band needs >=60s of data).
The other 8 features are valid at 30s.

Purpose: compare 30s vs 60s window length for the chest
multi-class pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.preprocessing.wesad_preprocessing import load_subject, create_chest_windows
from src.features.build_chest_features import build_chest_feature_matrix

SUBJECTS = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11",
            "S13","S14","S15","S16","S17"]
CLASS_NAMES = ["baseline", "stress", "amusement", "meditation"]
FLOOR = 0.01
WINDOW_SECONDS = 30

# hrv_lfhf is NaN at 30s -- drop it. The other 8 features are valid.
DROP_FEATURES = ["hrv_lfhf"]


def build_dataset():
    X_by_subj, y_by_subj = {}, {}
    for sid in SUBJECTS:
        data = load_subject(sid)
        ecg_w, resp_w, y = create_chest_windows(data, window_seconds=WINDOW_SECONDS)
        if len(y) == 0:
            continue
        X, names = build_chest_feature_matrix(ecg_w, resp_w)

        # Drop unstable features
        keep_idx = [i for i, n in enumerate(names) if n not in DROP_FEATURES]
        X = X[:, keep_idx]
        kept_names = [names[i] for i in keep_idx]

        # Drop windows with any remaining NaN
        valid = ~np.isnan(X).any(axis=1)
        X, y = X[valid], y[valid]

        X_by_subj[sid] = X
        y_by_subj[sid] = y

    print(f"Kept features: {kept_names}")
    return X_by_subj, y_by_subj


def calibrate_subject(X, y, floor=FLOOR):
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
        train_X, train_y = [], []
        for train_subj in subjects:
            if train_subj == test_subj:
                continue
            Xs, ys = X_by_subj[train_subj], y_by_subj[train_subj]
            if calibrate:
                Xs, _ = calibrate_subject(Xs, ys)
            train_X.append(Xs)
            train_y.append(ys)

        X_train = np.vstack(train_X)
        y_train = np.concatenate(train_y)
        X_test, y_test = X_by_subj[test_subj], y_by_subj[test_subj]
        if calibrate:
            X_test, _ = calibrate_subject(X_test, y_test)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                        class_weight="balanced",
                        decision_function_shape="ovr",
                        random_state=42)),
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
    print(f"CHEST MULTI-CLASS — {WINDOW_SECONDS}s WINDOWS (calibrated)")
    print("=" * 70)

    X_by_subj, y_by_subj = build_dataset()
    total = sum(len(y) for y in y_by_subj.values())
    print(f"\nSubjects: {len(X_by_subj)}")
    print(f"Total windows after NaN-drop: {total}")
    all_y = np.concatenate(list(y_by_subj.values()))
    print(f"Class distribution: {np.bincount(all_y, minlength=4)}")

    print("\n" + "-" * 70)
    print("WITH CALIBRATION")
    print("-" * 70)
    yt, yp, macro, per_class = run_loso(X_by_subj, y_by_subj, calibrate=True)

    print(classification_report(yt, yp, labels=[0, 1, 2, 3],
                                target_names=CLASS_NAMES,
                                zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(yt, yp, labels=[0, 1, 2, 3]))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Window: {WINDOW_SECONDS}s (features: 8, drop hrv_lfhf)")
    print(f"Macro F1: {macro:.4f}")
    print(f"Per-class F1: "
          f"baseline={per_class[0]:.4f}, "
          f"stress={per_class[1]:.4f}, "
          f"amusement={per_class[2]:.4f}, "
          f"meditation={per_class[3]:.4f}")

    print("\n" + "=" * 70)
    print("COMPARISON TO 60s WINDOWS (9 features, incl hrv_lfhf)")
    print("=" * 70)
    print(f"{'Window':<10}{'Features':<10}{'Macro F1':<12}"
          f"{'Baseline':<10}{'Stress':<10}{'Amuse':<10}{'Meditation':<10}")
    print("-" * 72)
    print(f"{'60s':<10}{'9':<10}{0.7774:<12.4f}{0.8610:<10.4f}"
          f"{0.8227:<10.4f}{0.5600:<10.4f}{0.8660:<10.4f}")
    print(f"{f'{WINDOW_SECONDS}s':<10}{'8':<10}{macro:<12.4f}"
          f"{per_class[0]:<10.4f}{per_class[1]:<10.4f}"
          f"{per_class[2]:<10.4f}{per_class[3]:<10.4f}")


if __name__ == "__main__":
    main()