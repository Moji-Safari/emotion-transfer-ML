import numpy as np

from src.features.eda_features import extract_eda_features


def build_feature_matrix(X, sampling_rate=4):
    """
    Convert all EDA windows into a numerical feature matrix.

    Parameters
    ----------
    X : numpy.ndarray
        Shape: (number_of_windows, samples_per_window)

    sampling_rate : int
        EDA sampling rate in Hz.

    Returns
    -------
    feature_matrix : numpy.ndarray
        Shape: (number_of_windows, number_of_features)
    """

    feature_rows = []

    for window in X:
        features = extract_eda_features(
            window,
            sampling_rate=sampling_rate
        )

        feature_rows.append(list(features.values()))

    return np.asarray(feature_rows, dtype=float)