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

def main():
    print("Building chest multi-class dataset...")

    all_X, all_y, all_g = [], [], []

    for sid in SUBJECTS:
        data = load_subject(sid)
        ecg_w, resp_w, y = create_chest_windows(data)

        if len(y) == 0:
            print(f"  {sid}: no valid windows")
            continue

        X, names = build_chest_feature_matrix(ecg_w, resp_w)

        # Drop windows with NaN
        valid = ~np.isnan(X).any(axis=1)
        X, y = X[valid], y[valid]

        print(f"  {sid}: {len(y)} windows, classes={np.bincount(y, minlength=4)}")

        all_X.append(X)
        all_y.append(y)
        all_g.append(np.full(len(y), sid, dtype=object))

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    groups = np.concatenate(all_g)

    print(f"\nTotal: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y, minlength=4)}")
    print(f"Feature names: {names}")

    # LOSO evaluation
    subjects = np.unique(groups)
    all_y_true, all_y_pred = [], []

    for test_subj in subjects:
        train_mask = groups != test_subj
        test_mask = groups == test_subj

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # RBF SVM with scaling inside fold
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                decision_function_shape="ovr",
                random_state=42,
            )),
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        print(f"  {test_subj}: F1_macro={f1_macro:.3f}")

    # Final report
    print("\n" + "=" * 70)
    print("MULTI-CLASS RESULTS (4 classes)")
    print("=" * 70)

    print("\nPer-class F1:")
    print(classification_report(
        all_y_true, all_y_pred,
        labels=[0, 1, 2, 3],
        target_names=["baseline", "stress", "amusement", "meditation"],
        zero_division=0,
    ))

    print("\nConfusion matrix:")
    print(confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2, 3]))

    print("\nMacro F1:", f1_score(all_y_true, all_y_pred, average="macro", zero_division=0))

if __name__ == "__main__":
    main()