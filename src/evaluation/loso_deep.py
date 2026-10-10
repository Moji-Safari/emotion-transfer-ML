"""
LOSO for the CNN-LSTM.

For each held-out subject:
    1. Build the training set from the other 14 subjects.
       Each training subject's EDA is calibrated using that
       subject's own baseline windows.
    2. Hold out a small validation split from the training set
       (stratified) for early stopping. Never touches the test
       subject.
    3. Calibrate the test subject's EDA using the test subject's
       own baseline windows.
    4. Train, predict, record metrics.

Calibration is per-subject and uses only baseline windows from
that same subject. No cross-subject leakage.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

from src.preprocessing.raw_windows import calibrate_eda_signal
from src.models.cnn_lstm_model import train_model, set_seeds


def _calibrate_group(X_group, y_group):
    """Apply per-subject calibration to one subject's windows."""
    return calibrate_eda_signal(X_group, y_group)[0]


def run_loso_deep(eda_windows_by_subject,
                  y_by_subject,
                  subject_ids,
                  epochs=100,
                  patience=10,
                  verbose=0):
    """
    Parameters
    ----------
    eda_windows_by_subject : dict[str, np.ndarray]
        Subject ID -> (n_windows, 120) raw EDA.
    y_by_subject : dict[str, np.ndarray]
        Subject ID -> (n_windows,) labels.
    subject_ids : list of str
    epochs, patience : int
    verbose : int
        Keras verbose level (0=silent, 1=progress).

    Returns
    -------
    results : list of dict
    y_true_all, y_pred_all : np.ndarray
    """
    results = []
    y_true_all, y_pred_all = [], []

    for test_subject in subject_ids:
        set_seeds()  # reproducibility per fold

        train_subjects = [s for s in subject_ids if s != test_subject]

        # --- Build calibrated training set ---
        X_train_parts, y_train_parts = [], []
        for s in train_subjects:
            Xs = eda_windows_by_subject[s]
            ys = y_by_subject[s]
            Xs_cal = _calibrate_group(Xs, ys)
            X_train_parts.append(Xs_cal)
            y_train_parts.append(ys)

        X_train_full = np.vstack(X_train_parts)
        y_train_full = np.concatenate(y_train_parts)

        # --- Stratified validation split (~15% of training) ---
        rng = np.random.RandomState(42)
        idx = np.arange(len(y_train_full))
        rng.shuffle(idx)
        val_frac = 0.15
        n_val = max(1, int(val_frac * len(idx)))

        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        X_train = X_train_full[train_idx]
        y_train = y_train_full[train_idx]
        X_val = X_train_full[val_idx]
        y_val = y_train_full[val_idx]

        # --- Build and train ---
        model = train_model(
            X_train, y_train, X_val, y_val,
            epochs=epochs, patience=patience, verbose=verbose,
        )

        # --- Calibrate and predict on test subject ---
        X_test_raw = eda_windows_by_subject[test_subject]
        y_test = y_by_subject[test_subject]
        X_test_cal = _calibrate_group(X_test_raw, y_test)

        X_test_r = X_test_cal[..., np.newaxis]
        y_prob = model.predict(X_test_r, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

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
            "confusion_matrix": cm,
        })

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        print(f"  {test_subject:<5} "
              f"n={len(y_test):<4}  "
              f"acc={acc:.3f}  f1={f1:.3f}")

    return results, np.asarray(y_true_all), np.asarray(y_pred_all)