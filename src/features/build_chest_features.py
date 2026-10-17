import numpy as np
from src.features.ecg_features import extract_ecg_features
from src.features.resp_features import extract_resp_features

def build_chest_feature_matrix(ecg_windows, resp_windows):
    """
    Build combined ECG + RESP feature matrix.
    """
    rows = []
    for ecg, resp in zip(ecg_windows, resp_windows):
        ecg_feats = extract_ecg_features(ecg)
        resp_feats = extract_resp_features(resp)
        row = list(ecg_feats.values()) + list(resp_feats.values())
        rows.append(row)

    return np.asarray(rows, dtype=float), list(ecg_feats.keys()) + list(resp_feats.keys())