"""
Build combined EDA + TEMP feature matrices.

Both signals are at 4 Hz, so windows align 1:1. Given the same
list of window start indices, we can extract EDA features and TEMP
features from the corresponding slices of each signal.

Feature order in the final matrix:
    [eda_mean, eda_std, eda_mean_abs_change,
     temp_mean, temp_std, temp_slope]

This order MUST be preserved for downstream code that references
columns by index.
"""

import numpy as np

from src.features.eda_features import extract_eda_features
from src.features.temp_features import extract_temp_features


# The three EDA features we keep (per reduced-feature experiment).
EDA_FEATURES_KEPT = ["mean", "std", "mean_absolute_change"]


def build_combined_feature_matrix(
    eda_windows,
    temp_windows,
    sampling_rate=4,
):
    """
    Extract EDA + TEMP features from aligned windows.

    Parameters
    ----------
    eda_windows : np.ndarray, shape (n_windows, samples_per_window)
    temp_windows : np.ndarray, shape (n_windows, samples_per_window)
        Must have the same shape as eda_windows.
    sampling_rate : int

    Returns
    -------
    feature_matrix : np.ndarray, shape (n_windows, 6)
    feature_names : list of str
        Column names, in order.
    """

    if eda_windows.shape != temp_windows.shape:
        raise ValueError(
            f"Window shapes differ: "
            f"EDA={eda_windows.shape}, TEMP={temp_windows.shape}"
        )

    rows = []
    for eda_w, temp_w in zip(eda_windows, temp_windows):
        eda_feats = extract_eda_features(eda_w, sampling_rate=sampling_rate)
        temp_feats = extract_temp_features(temp_w, sampling_rate=sampling_rate)

        row = [eda_feats[k] for k in EDA_FEATURES_KEPT]
        row += [temp_feats["temp_mean"],
                temp_feats["temp_std"],
                temp_feats["temp_slope"]]
        rows.append(row)

    feature_names = EDA_FEATURES_KEPT + [
        "temp_mean", "temp_std", "temp_slope"
    ]

    return np.asarray(rows, dtype=float), feature_names


def build_eda_only_matrix(eda_windows, sampling_rate=4):
    """Return just the 3 EDA features. For the EDA-only baseline."""
    rows = []
    for w in eda_windows:
        f = extract_eda_features(w, sampling_rate=sampling_rate)
        rows.append([f[k] for k in EDA_FEATURES_KEPT])
    return np.asarray(rows, dtype=float), list(EDA_FEATURES_KEPT)


def build_temp_only_matrix(temp_windows, sampling_rate=4):
    """Return just the 3 TEMP features. For the TEMP-only baseline."""
    rows = []
    for w in temp_windows:
        f = extract_temp_features(w, sampling_rate=sampling_rate)
        rows.append([f["temp_mean"], f["temp_std"], f["temp_slope"]])
    return np.asarray(rows, dtype=float), ["temp_mean", "temp_std", "temp_slope"]