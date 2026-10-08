"""
Peak-based features for wrist BVP (Blood Volume Pulse).

BVP is sampled at 64 Hz. Each 30-second window contains ~1920 samples
and ~30-40 heartbeats (at typical resting HR 60-80 bpm).

We use scipy.signal.find_peaks to detect beats, then compute:

    hr_mean              mean heart rate (bpm)
    hr_std               heart rate variability within the window
    pulse_amplitude_mean mean peak-to-trough amplitude of detected beats

These are the standard wrist-wearable stress markers derivable from
PPG. HRV features (RMSSD, SDNN) are not computed here — they require
accurate RR intervals and would be added in a follow-up if these
three features prove useful.

References
----------
Schmidt et al. (2018), "Introducing WESAD," ICMI 2018.
  https://doi.org/10.1145/3242969.3242985

Charlton et al. (2022), "Wearable Photoplethysmography for
Cardiovascular Monitoring." Frontiers in Digital Health.
  https://doi.org/10.3389/fdgth.2022.840840
"""

import numpy as np
from scipy.signal import find_peaks


BVP_FS = 64  # Hz


def extract_bvp_features(window, sampling_rate=BVP_FS):
    """
    Extract 3 peak-based features from one BVP window.

    Parameters
    ----------
    window : np.ndarray
        One BVP window (e.g. 1920 samples for 30 s at 64 Hz).

    sampling_rate : int
        BVP sampling rate in Hz. Should match the actual data.

    Returns
    -------
    dict with keys:
        bvp_hr_mean
        bvp_hr_std
        bvp_pulse_amplitude_mean

    If fewer than 3 peaks are detected (low-quality signal), returns
    NaN for all three features. The caller should handle NaN.
    """

    window = np.asarray(window, dtype=float).reshape(-1)

    # --- peak detection ---
    # min distance between peaks: assumes max HR = 180 bpm = 3 Hz
    # => at 64 Hz, 64/3 ≈ 21 samples between peaks
    min_distance = int(sampling_rate / 3.0)

    # min prominence: adaptive — 30% of the window's std, with floor
    prominence = max(0.3 * np.std(window), 1e-6)

    peaks, props = find_peaks(
        window,
        distance=min_distance,
        prominence=prominence,
    )

    if len(peaks) < 3:
        return {
            "bvp_hr_mean": np.nan,
            "bvp_hr_std":  np.nan,
            "bvp_pulse_amplitude_mean": np.nan,
        }

    # --- heart rate ---
    # RR intervals in seconds
    rr = np.diff(peaks) / sampling_rate

    # HR at each inter-beat interval (bpm)
    hr = 60.0 / rr

    # Exclude physiologically impossible HR (30-200 bpm)
    valid_hr = hr[(hr >= 30) & (hr <= 200)]

    if len(valid_hr) < 2:
        return {
            "bvp_hr_mean": np.nan,
            "bvp_hr_std":  np.nan,
            "bvp_pulse_amplitude_mean": np.nan,
        }

    # --- pulse amplitude ---
    # peak-to-trough: for each detected peak, find the preceding trough
    # Simple approach: local minima between consecutive peaks
    amplitudes = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        trough = np.min(window[start:end])
        amplitudes.append(window[peaks[i]] - trough)

    return {
        "bvp_hr_mean": float(np.mean(valid_hr)),
        "bvp_hr_std":  float(np.std(valid_hr)),
        "bvp_pulse_amplitude_mean": float(np.mean(amplitudes)),
    }