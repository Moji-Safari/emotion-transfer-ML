"""
Test whether stronger class weights improve amusement detection
in the chest multi-class pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from src.preprocessing.wesad_preprocessing import load_subject, create_chest_windows
from src.features.build_chest_features import build_chest_feature_matrix

SUBJECTS = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11",
            "S13","S14","S15","S16","S17"]
CLASS_NAMES = ["baseline", "stress", "amusement", "meditation"]


def build_dataset():
    all_X, all_y, all_g = [], [], []
    for sid in SUBJECTS:
        data = load_subject(sid)
        ecg_w, resp_w, y = create_chest_windows(data)
        if len(y) == 0:
            continue
        X, _ = build_chest_feature_matrix(ecg_w, resp_w)
        valid = ~np.isnan(X).any(axis=1)
        X, y = X[valid], y[valid]
        all_X.append(X)
        all_y.append(y)
        all_g.append(np.full(len(y), sid, dtype=object))
    return np.vstack(all_X), np.concatenate(all_y), np.concatenate(all_g)


def compute_weights(y_train, boost):
    classes = np.array([0, 1, 2, 3])
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weights = dict(zip(classes, w))
    weights[2] *= boost
    return weights


def run_condition(X, y, groups, boost, label):
    print(f"\n{'-' * 70}")
    print(f"CONDITION: {label}")
    print(f"{'-' * 70}")

    subjects = np.unique(groups)
    all_y_true, all_y_pred = [], []

    for test_subj in subjects:
        train_mask = groups != test_subj
        test_mask = groups == test_subj
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        cw = compute_weights(y_train, boost)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight=cw,
                decision_function_shape="ovr",
                random_state=42,
            )),
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    print(classification_report(all_y_true, all_y_pred,
                                labels=[0, 1, 2, 3],
                                target_names=CLASS_NAMES,
                                zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2, 3]))

    return {
        "label": label,
        "macro_f1": f1_score(all_y_true, all_y_pred, average="macro", zero_division=0),
        "amuse_f1": f1_score(all_y_true, all_y_pred, average=None,
                             labels=[0, 1, 2, 3], zero_division=0)[2],
    }


def main():
    print("=" * 70)
    print("CHEST CLASS WEIGHTING EXPERIMENT")
    print("=" * 70)

    X, y, groups = build_dataset()
    print(f"\nDataset: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y, minlength=4)}")

    results = [
        run_condition(X, y, groups, boost=1.0, label="balanced (baseline)"),
        run_condition(X, y, groups, boost=2.0, label="amusement × 2"),
        run_condition(X, y, groups, boost=4.0, label="amusement × 4"),
    ]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<24}{'Macro F1':<12}{'Amuse F1':<12}")
    print("-" * 48)
    for r in results:
        print(f"{r['label']:<24}{r['macro_f1']:<12.4f}"
              f"{r['amuse_f1']:<12.4f}")


if __name__ == "__main__":
    main()