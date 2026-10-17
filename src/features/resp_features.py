import numpy as np
import neurokit2 as nk
from scipy.signal import welch

def extract_resp_features(resp_window, sampling_rate=700):
    """
    Extract respiratory features from one 60s RESP window.
    """
    resp_window = np.asarray(resp_window, dtype=float).reshape(-1)

    # Clean and find peaks
    resp_clean = nk.rsp_clean(resp_window, sampling_rate=sampling_rate)
    peaks, info = nk.rsp_peaks(resp_clean, sampling_rate=sampling_rate)

    # Basic rate
    if len(info["RSP_Peaks"]) >= 3:
        rsp_rate = nk.rsp_rate(
            resp_clean, info, sampling_rate=sampling_rate
        )
        resp_rate_mean = float(np.mean(rsp_rate))
    else:
        resp_rate_mean = np.nan

    # Spectral energy in 0.7-1.0 Hz (stress marker)
    freqs, psd = welch(resp_clean, fs=sampling_rate, nperseg=min(4096, len(resp_clean)))
    band_mask = (freqs >= 0.7) & (freqs <= 1.0)
    if band_mask.sum() > 0:
        resp_spectral_energy = float(np.trapz(psd[band_mask], freqs[band_mask]))
    else:
        resp_spectral_energy = np.nan

    # Autocorrelation transition lag (meditation marker)
    # Simplified: first zero-crossing of autocorrelation
    resp_centered = resp_clean - np.mean(resp_clean)
    autocorr = np.correlate(resp_centered, resp_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]  # normalize
    zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
    if len(zero_crossings) > 0:
        resp_autocorr_lag = float(zero_crossings[0])
    else:
        resp_autocorr_lag = np.nan

    # Respiratory rate variability (RRV)
    if len(info["RSP_Peaks"]) >= 4:
        rrv = nk.rsp_rrv(rsp_rate, info, sampling_rate=sampling_rate)
        resp_rate_variability = float(rrv["RRV_RMSSD"].iloc[0])
    else:
        resp_rate_variability = np.nan

    return {
        "resp_rate_mean": resp_rate_mean,
        "resp_spectral_energy": resp_spectral_energy,
        "resp_autocorr_lag": resp_autocorr_lag,
        "resp_rate_variability": resp_rate_variability,
    }