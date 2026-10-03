import numpy as np


def extract_eda_features(window, sampling_rate=4):
    """
    Extract statistical features from one EDA window.

    Parameters
    ----------
    window : numpy.ndarray
        One EDA window.

    sampling_rate : int
        EDA sampling rate in Hz.

    Returns
    -------
    dict
        Extracted features.
    """

    window = np.asarray(window, dtype=float)

    features = {}

    # Basic statistical features
    features["mean"] = np.mean(window)
    features["std"] = np.std(window)
    features["min"] = np.min(window)
    features["max"] = np.max(window)
    features["range"] = np.max(window) - np.min(window)
    features["median"] = np.median(window)

    # Difference between consecutive samples
    differences = np.diff(window)

    features["mean_absolute_change"] = np.mean(
        np.abs(differences)
    )

    features["std_change"] = np.std(differences)

    # Linear trend
    time = np.arange(len(window)) / sampling_rate

    slope = np.polyfit(time, window, 1)[0]

    features["slope"] = slope

    return features