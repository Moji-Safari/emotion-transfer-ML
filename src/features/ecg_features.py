"""
ECG feature extraction using NeuroKit2.

Uses nk.hrv_time() and nk.hrv_frequency() separately instead of
nk.hrv() to avoid the DFA computation that fails on short windows.

HRV total power is read from the HRV_TP column and converted
from seconds^2 to ms^2 by multiplying by 1e6.
"""

import numpy as np
import neurokit2 as nk


def extract_ecg_features(ecg_window, sampling_rate=700):
    """
    Extract 5 HRV features from one 60-second ECG window.

    Parameters
    ----------
    ecg_window : np.ndarray
        One ECG window (60s * 700Hz = 42000 samples).
    sampling_rate : int
        ECG sampling rate. WESAD chest ECG is 700 Hz.

    Returns
    -------
    dict with keys:
        ecg_hr_mean, hrv_rmssd, hrv_sdnn, hrv_lfhf, hrv_total_power
    """
    ecg_window = np.asarray(ecg_window, dtype=float).reshape(-1)

    try:
        ecg_clean = nk.ecg_clean(ecg_window, sampling_rate=sampling_rate)
    except Exception:
        return _nan_features()

    try:
        peaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate)
    except Exception:
        return _nan_features()

    rpeaks = info.get("ECG_R_Peaks", [])
    if len(rpeaks) < 5:
        return _nan_features()

    # Mean heart rate from RR intervals
    rr_seconds = np.diff(rpeaks) / sampling_rate
    hr_mean = float(np.mean(60.0 / rr_seconds)) if len(rr_seconds) > 0 else np.nan

    # Time-domain and frequency-domain HRV
    rpeaks_dict = {"ECG_R_Peaks": rpeaks}

    try:
        hrv_time_df = nk.hrv_time(
            rpeaks_dict, sampling_rate=sampling_rate, show=False
        )
    except Exception:
        hrv_time_df = None

    try:
        hrv_freq_df = nk.hrv_frequency(
            rpeaks_dict, sampling_rate=sampling_rate, show=False
        )
    except Exception:
        hrv_freq_df = None

    def safe_get(df, col):
        if df is None:
            return np.nan
        try:
            return float(df[col].iloc[0])
        except (KeyError, IndexError):
            return np.nan

    # Total power: HRV_TP is in seconds^2. Convert to ms^2.
    tp_seconds2 = safe_get(hrv_freq_df, "HRV_TP")
    total_power = float(tp_seconds2 * 1e6) if not np.isnan(tp_seconds2) else np.nan

    return {
        "ecg_hr_mean": hr_mean,
        "hrv_rmssd": safe_get(hrv_time_df, "HRV_RMSSD"),
        "hrv_sdnn": safe_get(hrv_time_df, "HRV_SDNN"),
        "hrv_lfhf": safe_get(hrv_freq_df, "HRV_LFHF"),
        "hrv_total_power": total_power,
    }


def _nan_features():
    return {
        "ecg_hr_mean": np.nan,
        "hrv_rmssd": np.nan,
        "hrv_sdnn": np.nan,
        "hrv_lfhf": np.nan,
        "hrv_total_power": np.nan,
    }