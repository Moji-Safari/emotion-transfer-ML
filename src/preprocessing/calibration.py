"""
Calibration-aware feature transformation.

Idea
----
Different subjects have different baseline physiology. A subject with
naturally high EDA (say, 2.0 µS) at rest will look "stressed" to a
model trained on subjects who rest at 0.3 µS. Conversely, a subject
with low reactivity (S14 in our data) will look "not stressed" even
during stress, because their absolute EDA values overlap with the
population baseline.

Calibration removes the subject-specific offset by transforming each
subject's features into *relative deviations* from their own baseline:

    X* = (X - B_s) / (|B_s| + epsilon)

where B_s is the mean feature vector over the subject's own baseline
windows.

This is applied per subject, using only that subject's baseline data.
No information from other subjects leaks in.

Reference
---------
Personalized Electrocardiographic and HRV Dynamics for Acute Stress
Detection: A Leave-One-Subject-Out Benchmark on WESAD.
Zenodo Preprint, 2026.
https://zenodo.org/records/22806710
"""

import numpy as np


class CalibrationTransformer:
    """
    Per-subject feature calibration.

    Usage
    -----
    ct = CalibrationTransformer()
    ct.fit(baseline_windows)   # baseline_windows.shape = (n, n_features)
    X_calibrated = ct.transform(X)  # any windows from the same subject
    """

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.baseline_mean = None
        self.baseline_scale = None

    def fit(self, baseline_windows):
        """
        Compute the calibration statistics from the subject's own
        baseline windows.

        Parameters
        ----------
        baseline_windows : np.ndarray, shape (n, d)
            Feature vectors from windows known to be in the baseline
            condition for this subject. Must be non-empty.
        """
        baseline_windows = np.asarray(baseline_windows, dtype=float)
        if baseline_windows.ndim != 2 or baseline_windows.shape[0] < 1:
            raise ValueError(
                f"baseline_windows must be 2D with >= 1 row, "
                f"got shape {baseline_windows.shape}"
            )

        self.baseline_mean = baseline_windows.mean(axis=0)
        # Use absolute value as scale so the sign of the shift is preserved
        self.baseline_scale = np.abs(self.baseline_mean) + self.epsilon
        return self

    def transform(self, X):
        """Apply calibration: X* = (X - B_s) / |B_s|."""
        if self.baseline_mean is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=float)
        return (X - self.baseline_mean) / self.baseline_scale

class CalibrationTransformerStd:
    """
    Per-subject feature calibration using baseline standard
    deviation as the scale factor.

    X* = (X - B_s) / (sigma_s + epsilon)

    where B_s is the mean of the baseline windows and sigma_s is
    their standard deviation (per feature).

    Compared to CalibrationTransformer (which divides by |B_s|),
    this produces features with approximately unit variance over
    the baseline period, regardless of the feature's absolute
    units.

    Motivation: temperature is ~30°C while EDA is ~0.3 µS.
    Dividing by |B_s| produces a ~100x scale difference between
    features, which the RBF SVM's distance metric may not handle
    well. Dividing by sigma_s equalizes scales.
    """

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.baseline_mean = None
        self.baseline_std = None

    def fit(self, baseline_windows):
        baseline_windows = np.asarray(baseline_windows, dtype=float)
        if baseline_windows.ndim != 2 or baseline_windows.shape[0] < 2:
            raise ValueError(
                f"baseline_windows must be 2D with >= 2 rows, "
                f"got shape {baseline_windows.shape}"
            )
        self.baseline_mean = baseline_windows.mean(axis=0)
        self.baseline_std = baseline_windows.std(axis=0)
        return self

    def transform(self, X):
        if self.baseline_mean is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=float)
        return (X - self.baseline_mean) / (self.baseline_std + self.epsilon)


def calibrate_subject_std(X_subject, y_subject):
    """
    Same interface as calibrate_subject(), but uses std-based
    calibration.
    """
    baseline_mask = (y_subject == 0)
    if baseline_mask.sum() < 2:
        return X_subject.copy(), False

    ct = CalibrationTransformerStd()
    ct.fit(X_subject[baseline_mask])
    return ct.transform(X_subject), True


def calibrate_subject(X_subject, y_subject):
    """
    Calibrate one subject's windows using their own baseline.

    Parameters
    ----------
    X_subject : np.ndarray, shape (n, d)
    y_subject : np.ndarray, shape (n,)
        0 = baseline, 1 = stress.

    Returns
    -------
    X_calibrated : np.ndarray, shape (n, d)
        Same shape. Calibrated values.
    ok : bool
        True if calibration could be applied (baseline windows
        existed). False if the subject has no baseline windows, in
        which case X_calibrated == X_subject.
    """
    baseline_mask = (y_subject == 0)

    if baseline_mask.sum() < 1:
        # No baseline data — cannot calibrate this subject.
        return X_subject.copy(), False

    ct = CalibrationTransformer()
    ct.fit(X_subject[baseline_mask])
    return ct.transform(X_subject), True