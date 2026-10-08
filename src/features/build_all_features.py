"""
Combined feature matrix builder: EDA + TEMP + BVP.

Feature set v3 = 4 features:
    eda_mean, eda_std, eda_mean_absolute_change, temp_mean

Feature set v4 (this file) adds BVP:
    + bvp_hr_mean, bvp_hr_std, bvp_pulse_amplitude_mean

Total: 7 features.

Windows with NaN BVP features are dropped (see extract_bvp_features).
The function reports how many windows were dropped.
"""

import numpy as np

from src.features.eda_features import extract_eda_features
from src.features.temp_features import extract_temp_features
from src.features.bvp_features import extract_bvp_features


EDA_KEEP = ["mean", "std", "mean_absolute_change"]
EDA_TEMP_FEATURES = EDA_KEEP + ["temp_mean"]
EDA_TEMP_BVP_FEATURES = EDA_TEMP_FEATURES + [
    "bvp_hr_mean", "bvp_hr_std", "bvp_pulse_amplitude_mean",
]


def build_eda_temp_matrix(eda_windows, temp_windows):
    """Feature set v3. 4 features."""
    rows = []
    for e, t in zip(eda_windows, temp_windows):
        ef = extract_eda_features(e, sampling_rate=4)
        tf = extract_temp_features(t, sampling_rate=4)
        row = [ef[k] for k in EDA_KEEP] + [tf["temp_mean"]]
        rows.append(row)
    return np.asarray(rows, dtype=float), list(EDA_TEMP_FEATURES)


def build_eda_temp_bvp_matrix(eda_windows, temp_windows, bvp_windows):
    """
    Feature set v4. 7 features.

    Drops windows where BVP features are NaN (peak detection failed).
    Returns (matrix, feature_names, kept_mask) so the caller can
    align labels.
    """
    rows = []
    kept = []
    n_dropped = 0

    for e, t, b in zip(eda_windows, temp_windows, bvp_windows):
        ef = extract_eda_features(e, sampling_rate=4)
        tf = extract_temp_features(t, sampling_rate=4)
        bf = extract_bvp_features(b, sampling_rate=64)

        row = [ef[k] for k in EDA_KEEP] + [tf["temp_mean"]]
        row += [
            bf["bvp_hr_mean"],
            bf["bvp_hr_std"],
            bf["bvp_pulse_amplitude_mean"],
        ]

        if any(np.isnan(v) for v in row):
            n_dropped += 1
            kept.append(False)
        else:
            rows.append(row)
            kept.append(True)

    if n_dropped > 0:
        print(f"  [build_eda_temp_bvp_matrix] "
              f"dropped {n_dropped}/{len(eda_windows)} windows "
              f"(BVP peak detection failed)")

    return (
        np.asarray(rows, dtype=float),
        list(EDA_TEMP_BVP_FEATURES),
        np.asarray(kept, dtype=bool),
    )