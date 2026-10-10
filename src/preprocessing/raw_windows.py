"""
Calibrated raw EDA windows for deep learning.

Each window is 120 EDA samples (30 s at 4 Hz), transformed using
per-subject calibration:

    x_calibrated = (x_raw - B_s) / (sigma_s + epsilon)

where B_s and sigma_s are computed from the subject's own baseline
windows.

For DL we also need a floor on sigma_s to prevent numerical
instability on subjects with unusually stable baselines
(analogous to the feature-level floor).
"""

import numpy as np


SIGMA_FLOOR = 0.01  # same value as feature-level calibration


def calibrate_eda_signal(eda_windows, labels):
    """
    Calibrate raw EDA windows using per-subject baseline statistics.

    Parameters
    ----------
    eda_windows : np.ndarray, shape (n_windows, 120)
        Raw EDA values.
    labels : np.ndarray, shape (n_windows,)
        0 = baseline, 1 = stress.

    Returns
    -------
    calibrated : np.ndarray, shape (n_windows, 120)
        Calibrated EDA windows.
    B_s : float
        Baseline mean (all baseline samples pooled).
    sigma_s : float
        Baseline std (all baseline samples pooled), clipped to
        SIGMA_FLOOR.
    """
    baseline_mask = (labels == 0)
    if baseline_mask.sum() == 0:
        # No baseline: return uncalibrated
        return eda_windows.copy(), 0.0, 1.0

    # Pool all baseline samples into one flat vector
    baseline_samples = eda_windows[baseline_mask].reshape(-1)
    B_s = float(baseline_samples.mean())
    sigma_s = float(baseline_samples.std())
    sigma_s = max(sigma_s, SIGMA_FLOOR)

    calibrated = (eda_windows - B_s) / sigma_s
    return calibrated, B_s, sigma_s