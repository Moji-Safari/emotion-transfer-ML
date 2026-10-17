import numpy as np
import neurokit2 as nk

def extract_ecg_features(ecg_window, sampling_rate=700):
    """
    Extract HRV features from one 60s ECG window.

    Uses NeuroKit2's interval-related analysis.
    """
    ecg_window = np.asarray(ecg_window, dtype=float).reshape(-1)

    # Clean and find R-peaks
    ecg_clean = nk.ecg_clean(ecg_window, sampling_rate=sampling_rate)
    peaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate)

    if len(info["ECG_R_Peaks"]) < 5:
        return {
            "ecg_hr_mean": np.nan,
            "hrv_rmssd": np.nan,
            "hrv_sdnn": np.nan,
            "hrv_lfhf": np.nan,
            "hrv_total_power": np.nan,
        }

    # Interval-related features
    interval = nk.ecg_intervalrelated(
        {"ECG_R_Peaks": info["ECG_R_Peaks"]},
        sampling_rate=sampling_rate,
    )

    def safe_get(df, col):
        try:
            return float(df[col].iloc[0])
        except (KeyError, IndexError):
            return np.nan

    return {
        "ecg_hr_mean": safe_get(interval, "ECG_Rate_Mean"),
        "hrv_rmssd": safe_get(interval, "HRV_RMSSD"),
        "hrv_sdnn": safe_get(interval, "HRV_SDNN"),
        "hrv_lfhf": safe_get(interval, "HRV_LFHF"),
        "hrv_total_power": safe_get(interval, "HRV_Total_Power"),
    }