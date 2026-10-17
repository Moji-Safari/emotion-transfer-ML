"""
RESP feature extraction using NeuroKit2.

Computes respiratory rate, spectral energy in the 0.70-1.00 Hz
band (stress marker), autocorrelation lag (meditation marker),
and RRV (RMSSD of breath intervals).

Uses np.trapezoid (NumPy 2.x); falls back to np.trapz on NumPy 1.x.
Downsamples to 50 Hz before PSD, matching the 2026 paper on
state-specific respiratory signatures.
"""

import numpy as np
import neurokit2 as nk
from scipy.signal import welch, decimate


# NumPy 2.x renamed trapz -> trapezoid. Handle both.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def extract_resp_features(resp_window, sampling_rate=700):
    """
    Extract 4 respiratory features from one 60-second RESP window.

    Parameters
    ----------
    resp_window : np.ndarray
        One RESP window (60s * 700Hz = 42000 samples).
    sampling_rate : int
        RESP sampling rate. WESAD chest RESP is 700 Hz.

    Returns
    -------
    dict with keys:
        resp_rate_mean
        resp_spectral_energy   (0.70-1.00 Hz band, stress marker)
        resp_autocorr_lag      (meditation marker)
        resp_rate_variability  (RMSSD of breath intervals)
    """
    resp_window = np.asarray(resp_window, dtype=float).reshape(-1)

    try:
        resp_clean = nk.rsp_clean(resp_window, sampling_rate=sampling_rate)
    except Exception:
        return _nan_features()

    try:
        peaks, info = nk.rsp_peaks(resp_clean, sampling_rate=sampling_rate)
    except Exception:
        return _nan_features()

    rsp_peaks = info.get("RSP_Peaks", [])
    if len(rsp_peaks) < 3:
        return _nan_features()

    # Mean respiratory rate (breaths/min)
    breath_intervals_sec = np.diff(rsp_peaks) / sampling_rate
    if len(breath_intervals_sec) > 0:
        resp_rate_mean = float(np.mean(60.0 / breath_intervals_sec))
    else:
        resp_rate_mean = np.nan

    # Spectral energy in 0.70-1.00 Hz band (stress marker)
    try:
        q = max(1, sampling_rate // 50)
        resp_50 = decimate(resp_clean, q=q, zero_phase=True)
        freqs, psd = welch(resp_50, fs=50,
                           nperseg=min(512, len(resp_50)))
        band_mask = (freqs >= 0.70) & (freqs <= 1.00)
        if band_mask.sum() > 0:
            resp_spectral_energy = float(
                _trapz(psd[band_mask], freqs[band_mask])
            )
        else:
            resp_spectral_energy = np.nan
    except Exception:
        resp_spectral_energy = np.nan

    # Autocorrelation first zero-crossing (meditation marker)
    try:
        centered = resp_clean - np.mean(resp_clean)
        max_lag = min(len(centered) // 2, sampling_rate * 30)
        centered = centered[: 2 * max_lag]

        autocorr = np.correlate(centered, centered, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        if autocorr[0] != 0:
            autocorr = autocorr / autocorr[0]

        zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
        if len(zero_crossings) > 0:
            resp_autocorr_lag = float(zero_crossings[0])
        else:
            resp_autocorr_lag = np.nan
    except Exception:
        resp_autocorr_lag = np.nan

    # RRV (RMSSD of breath intervals) -- manual
    if len(breath_intervals_sec) > 2:
        diffs = np.diff(breath_intervals_sec)
        resp_rate_variability = float(np.sqrt(np.mean(diffs ** 2)))
    else:
        resp_rate_variability = np.nan

    return {
        "resp_rate_mean": resp_rate_mean,
        "resp_spectral_energy": resp_spectral_energy,
        "resp_autocorr_lag": resp_autocorr_lag,
        "resp_rate_variability": resp_rate_variability,
    }


def _nan_features():
    return {
        "resp_rate_mean": np.nan,
        "resp_spectral_energy": np.nan,
        "resp_autocorr_lag": np.nan,
        "resp_rate_variability": np.nan,
    }