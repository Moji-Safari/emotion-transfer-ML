"""
LOSO with std-based calibration, with a floor on sigma_s.

Same structure as loso_calibrated_std.py, but passes a floor
parameter through to calibrate_subject_std_floored().
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

from src.models.model_factory import MODEL_REGISTRY
from src.preprocessing.calibration import calibrate_subject_std_floored


def run_loso_calibrated_std_floored(X, y, groups,
                                    model_name="svm_rbf",
                                    floor=0.01,
                                    verbose=True):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'")

    make_model = MODEL_REGISTRY[model_name]
    subjects = np.unique(groups)

    results = []
    y_true_all, y_pred_all = [], []

    for test_subject in subjects:
        X_train_parts, y_train_parts = [], []
        for train_subject in subjects:
            if train_subject == test_subject:
                continue
            mask = groups == train_subject
            Xs = X[mask]
            ys = y[mask]
            Xs_cal, ok = calibrate_subject_std_floored(Xs, ys,
                                                       floor=floor)
            X_train_parts.append(Xs_cal)
            y_train_parts.append(ys)

        X_train = np.vstack(X_train_parts)
        y_train = np.concatenate(y_train_parts)

        test_mask = groups == test_subject
        X_test_raw = X[test_mask]
        y_test = y[test_mask]

        X_test, ok = calibrate_subject_std_floored(X_test_raw, y_test,
                                                   floor=floor)

        model = make_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        results.append({
            "subject": test_subject,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "n_test": int(len(y_test)),
            "n_test_baseline": int(np.sum(y_test == 0)),
            "n_test_stress": int(np.sum(y_test == 1)),
            "confusion_matrix": cm,
            "calibrated": ok,
        })

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        if verbose:
            print(f"  {str(test_subject):<5} "
                  f"n={len(y_test):<4} "
                  f"acc={acc:.3f}  f1={f1:.3f}")

    return results, np.asarray(y_true_all), np.asarray(y_pred_all)