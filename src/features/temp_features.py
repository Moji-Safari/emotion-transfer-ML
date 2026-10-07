"""
Statistical features for wrist skin temperature (TEMP).

TEMP is sampled at 4 Hz (same as EDA), so windows align naturally.

Design
------
We mirror the EDA feature design but with fewer features, since TEMP
is known to be less informative for stress detection than EDA.
Starting small: 3 features, one per concept (level, dispersion, trend).

References
----------
Schmidt et al. (2018), "Introducing WESAD," ICMI 2018.
  https://doi.org/10.1145/3242969.3242985

Can et al. (2019), "Continuous Stress Detection Using Wearable Sensors
in Real Life," Sensors, 19(8), 1849.
  https://www.mdpi.com/1424-8220/19/8/1849
"""

import numpy as np


def extract_temp_features(window, sampling_rate=4):
    """
    Extract 3 statistical features from one TEMP window.

    Parameters
    ----------
    window : np.ndarray
        One TEMP window (e.g. 120 samples for 30 s at 4 Hz).

    sampling_rate : int
        Sampling rate in Hz.

    Returns
    -------
    dict
        Feature name -> value, in this exact order:
        mean, std, slope
    """

    window = np.asarray(window, dtype=float)

    features = {}

    # Central level
    features["temp_mean"] = float(np.mean(window))

    # Dispersion
    features["temp_std"] = float(np.std(window))

    # Temporal trend
    time = np.arange(len(window)) / sampling_rate
    features["temp_slope"] = float(np.polyfit(time, window, 1)[0])

    return features