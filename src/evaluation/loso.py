"""
Leave-One-Subject-Out evaluation.

For each subject S:
    - train on all subjects != S
    - test on S
    - record metrics

No subject appears in both training and test for any fold.
No scaler is fit on test-fold data. No threshold is tuned on
test-fold data. The evaluation is as honest as we can make it.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.models.model_factory import MODEL_REGISTRY


def run_loso(X, y, groups, model_name="random_forest", verbose=True):
    """
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    groups : np.ndarray, shape (n_samples,)
        Subject identifier per sample.
    model_name : str
        Key in MODEL_REGISTRY.
    verbose : bool
        Print per-fold progress.

    Returns
    -------
    results : list of dict
        Per-subject metrics and confusion matrix.
    y_true_all : np.ndarray
        Concatenated ground-truth labels across all folds.
    y_pred_all : np.ndarray
        Concatenated predictions across all folds.
    """

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY)}"
        )

    make_model = MODEL_REGISTRY[model_name]
    subjects = np.unique(groups)

    results = []
    y_true_all, y_pred_all = [], []

    for test_subject in subjects:
        train_mask = groups != test_subject
        test_mask = groups == test_subject

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

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
        })

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        if verbose:
            print(
                f"  {str(test_subject):<5} "
                f"n={len(y_test):<4} "
                f"base={np.sum(y_test == 0):<4} "
                f"stress={np.sum(y_test == 1):<4} "
                f"acc={acc:.3f}  "
                f"f1={f1:.3f}"
            )

    return results, np.asarray(y_true_all), np.asarray(y_pred_all)