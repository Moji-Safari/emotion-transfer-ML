"""
Flask API for AffectWave.

Endpoints
---------
GET  /health
    Server status and model metadata.

POST /calibrate
    Compute per-subject calibration statistics (B_s, sigma_s)
    from baseline feature windows.

POST /predict
    Classify one feature window using the wrist binary SVM.

POST /predict-from-conditions
    Accept simple body-condition inputs, derive the 4 wrist
    features by a documented heuristic, and classify with the
    wrist binary SVM.

POST /predict-affective-state
    Accept 7 body-condition inputs, derive the 9 chest features
    by a documented heuristic, and classify with the four-class
    chest SVM.

POST /generate-haptic
    Map a stress probability to a vibrotactile pattern.

Run:
    python -m api.app
"""

import json
import pickle
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from api.haptic_mapping import stress_to_haptic


MODEL_DIR = Path("models/deployed")
FLOOR = 0.01

app = Flask(__name__)
CORS(app)


# ------------------------------------------------------------------
# Load models at startup
# ------------------------------------------------------------------
_wrist_model_path = MODEL_DIR / "svm.pkl"
_wrist_names_path = MODEL_DIR / "feature_names.json"

if not _wrist_model_path.exists():
    raise FileNotFoundError(
        f"Wrist model not found at {_wrist_model_path}. "
        f"Run scripts.export_model first."
    )

with open(_wrist_model_path, "rb") as f:
    model = pickle.load(f)
with open(_wrist_names_path) as f:
    FEATURE_NAMES = json.load(f)
print(f"Loaded wrist binary model. Features: {FEATURE_NAMES}")


_chest_model_path = MODEL_DIR / "chest_svm.pkl"
_chest_names_path = MODEL_DIR / "chest_feature_names.json"

if _chest_model_path.exists() and _chest_names_path.exists():
    with open(_chest_model_path, "rb") as f:
        chest_model = pickle.load(f)
    with open(_chest_names_path) as f:
        CHEST_FEATURE_NAMES = json.load(f)
    print(f"Loaded chest 4-class model. Features: {CHEST_FEATURE_NAMES}")
else:
    chest_model = None
    CHEST_FEATURE_NAMES = []
    print("Warning: chest 4-class model not found.")


CHEST_STATES = ["baseline", "stress", "amusement", "meditation"]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _calibrate(X, B_s, sigma_s):
    denom = np.maximum(sigma_s, FLOOR)
    return (X - B_s) / denom


def _validate_wrist_features(features):
    arr = np.asarray(features, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"Expected 4 wrist features, got shape {arr.shape}")
    return arr.reshape(1, -1)


def _wrist_proba(X):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0, 1])
    score = float(model.decision_function(X)[0])
    return float(1.0 / (1.0 + np.exp(-score)))


def _wrist_band(p):
    if p < 0.35:
        return "low"
    if p < 0.65:
        return "medium"
    return "high"


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_dir": str(MODEL_DIR.resolve()),
        "wrist_features": FEATURE_NAMES,
        "chest_features": CHEST_FEATURE_NAMES,
        "wrist_model_loaded": True,
        "chest_model_loaded": chest_model is not None,
    })


@app.route("/calibrate", methods=["POST"])
def calibrate():
    data = request.get_json(silent=True) or {}
    baseline = data.get("baseline_features")

    if baseline is None:
        return jsonify({"error": "missing 'baseline_features'"}), 400

    arr = np.asarray(baseline, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        return jsonify({
            "error": f"baseline_features must be (n, 4), got {arr.shape}"
        }), 400
    if arr.shape[0] < 2:
        return jsonify({"error": "need at least 2 baseline windows"}), 400

    B_s = arr.mean(axis=0).tolist()
    sigma_s = arr.std(axis=0).tolist()

    return jsonify({
        "B_s": B_s,
        "sigma_s": sigma_s,
        "n_baseline_windows": int(arr.shape[0]),
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    try:
        X_raw = _validate_wrist_features(data.get("features"))
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    cal = data.get("calibration")
    if cal is not None:
        try:
            B_s = np.asarray(cal["B_s"], dtype=float)
            sigma_s = np.asarray(cal["sigma_s"], dtype=float)
            if B_s.shape != (4,) or sigma_s.shape != (4,):
                raise ValueError("B_s and sigma_s must have length 4")
            X = _calibrate(X_raw, B_s, sigma_s)
        except (KeyError, ValueError, TypeError) as e:
            return jsonify({"error": f"invalid calibration: {e}"}), 400
    else:
        X = X_raw

    proba = _wrist_proba(X)
    band = _wrist_band(proba)

    return jsonify({
        "stress_probability": round(proba, 4),
        "prediction": int(proba >= 0.5),
        "band": band,
        "calibrated": cal is not None,
    })


@app.route("/predict-from-conditions", methods=["POST"])
def predict_from_conditions():
    data = request.get_json(silent=True) or {}

    try:
        sweating = float(data["sweating"])
        skin_temp = float(data["skin_temp_c"])
        eda_baseline = float(data["eda_baseline"])
        heart_rate = float(data["heart_rate"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"invalid body conditions: {e}"}), 400

    sweating = max(0.0, min(1.0, sweating))
    skin_temp = max(28.0, min(36.0, skin_temp))
    eda_baseline = max(0.05, min(2.0, eda_baseline))
    heart_rate = max(50.0, min(120.0, heart_rate))

    eda_mean = eda_baseline + sweating * 3.0
    eda_std = 0.02 + sweating * 0.15 + (heart_rate - 70) / 500.0
    eda_mac = 0.005 + sweating * 0.03 + (heart_rate - 70) / 2000.0
    temp_mean = skin_temp

    features = np.array([[eda_mean, eda_std, eda_mac, temp_mean]])
    proba = _wrist_proba(features)
    band = _wrist_band(proba)

    return jsonify({
        "stress_probability": round(proba, 4),
        "prediction": int(proba >= 0.5),
        "band": band,
        "derived_features": {
            "eda_mean": round(eda_mean, 4),
            "eda_std": round(eda_std, 4),
            "eda_mean_absolute_change": round(eda_mac, 4),
            "temp_mean": round(temp_mean, 4),
        },
        "note": (
            "Features derived from body conditions by a documented "
            "heuristic, then evaluated by the trained wrist SVM."
        ),
    })

# Baseline conditions the API uses for calibration.
# These match DEFAULT_CONDITIONS in the frontend.
API_BASELINE_CONDITIONS = {
    "heart_rate": 72.0,
    "hrv_rmssd": 42.0,
    "hrv_sdnn": 65.0,
    "respiratory_rate": 16.0,
    "respiratory_energy": 0.02,
    "respiratory_variability": 0.5,
    "skin_temp_c": 33.0,
}


def _conditions_to_chest_features(conditions):
    """
    Convert 7 body conditions into the 9 chest features the model
    expects. Same heuristic used in the API endpoint.
    """
    hr = conditions["heart_rate"]
    rmssd = conditions["hrv_rmssd"]
    sdnn = conditions["hrv_sdnn"]
    rr = conditions["respiratory_rate"]
    re = conditions["respiratory_energy"]
    rv = conditions["respiratory_variability"]
    temp = conditions["skin_temp_c"]

    lfhf = 0.3 + (hr - 60.0) / 10.0 + (60.0 - rmssd) / 20.0
    lfhf = max(0.3, min(8.0, lfhf))

    tp = max(20000.0, min(150000.0, rmssd * 1500.0))

    autocorr_lag = max(300.0, min(800.0, 300.0 + rr * 20.0))

    return np.array([
        hr, rmssd, sdnn, lfhf, tp,
        rr, re, autocorr_lag, rv,
    ], dtype=float)



@app.route("/generate-haptic", methods=["POST"])
def generate_haptic():
    data = request.get_json(silent=True) or {}
    p = data.get("stress_probability")
    if p is None:
        return jsonify({"error": "missing 'stress_probability'"}), 400
    try:
        pattern = stress_to_haptic(float(p))
    except (TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(pattern)

def _conditions_to_chest_features(conditions):
    """Convert one set of body conditions to the 9 chest features."""
    hr = float(conditions["heart_rate"])
    rmssd = float(conditions["hrv_rmssd"])
    sdnn = float(conditions["hrv_sdnn"])
    rr = float(conditions["respiratory_rate"])
    re = float(conditions["respiratory_energy"])
    rv = float(conditions["respiratory_variability"])

    lfhf = 0.3 + (hr - 60.0) / 10.0 + (60.0 - rmssd) / 20.0
    lfhf = max(0.3, min(8.0, lfhf))
    tp = max(20000.0, min(150000.0, rmssd * 1500.0))
    autocorr_lag = max(300.0, min(800.0, 300.0 + rr * 20.0))

    return np.array([
        hr, rmssd, sdnn, lfhf, tp,
        rr, re, autocorr_lag, rv,
    ], dtype=float)

@app.route("/predict-affective-state", methods=["POST"])
def predict_affective_state():
    if chest_model is None:
        return jsonify({"error": "chest model not loaded"}), 503

    data = request.get_json(silent=True) or {}

    baseline_list = data.get("baseline")
    current_conditions = data.get("current")

    if baseline_list is None or current_conditions is None:
        return jsonify({
            "error": "request must include 'baseline' (list) "
                     "and 'current' (object)"
        }), 400

    if not isinstance(baseline_list, list) or len(baseline_list) < 5:
        return jsonify({
            "error": "baseline must be a list of at least 5 condition dicts"
        }), 400

    try:
        baseline_X = np.vstack([
            _conditions_to_chest_features(c) for c in baseline_list
        ])
        current_X = _conditions_to_chest_features(current_conditions)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"invalid conditions: {e}"}), 400

    # Compute calibration statistics with realistic physiological
    # sigma floors. Without the floors, jittered baseline readings
    # produce near-zero sigma, and every current input is treated
    # as many standard deviations from the baseline.
    SIGMA_FLOORS = {
        "ecg_hr_mean": 3.0,
        "hrv_rmssd": 8.0,
        "hrv_sdnn": 10.0,
        "hrv_lfhf": 0.5,
        "hrv_total_power": 15000.0,
        "resp_rate_mean": 1.5,
        "resp_spectral_energy": 0.01,
        "resp_autocorr_lag": 50.0,
        "resp_rate_variability": 0.15,
    }
    floors = np.array([SIGMA_FLOORS[n] for n in CHEST_FEATURE_NAMES])

    B_s = baseline_X.mean(axis=0)
    sigma_s = baseline_X.std(axis=0)
    sigma_s = np.maximum(sigma_s, floors)

    current_cal = (current_X - B_s) / sigma_s
    X_input = current_cal.reshape(1, -1)

    scores = chest_model.decision_function(X_input)[0]
    exp = np.exp(scores - np.max(scores))
    proba = exp / exp.sum()
    pred_idx = int(np.argmax(proba))

    pred_state = CHEST_STATES[pred_idx]
    confidence = float(proba[pred_idx])

    HAPTIC_HZ = {
        "baseline":   0.3,
        "meditation": 0.5,
        "amusement":  2.0,
        "stress":     6.0,
    }

    return jsonify({
        "state": pred_state,
        "confidence": round(confidence, 4),
        "probabilities": {
            CHEST_STATES[i]: round(float(proba[i]), 4)
            for i in range(len(CHEST_STATES))
        },
        "haptic_hz": HAPTIC_HZ.get(pred_state, 0.3),
        "calibration": {
            "n_baseline": len(baseline_list),
            "B_s": [round(float(x), 4) for x in B_s],
            "sigma_s": [round(float(x), 4) for x in sigma_s],
        },
        "current_calibrated": {
            name: round(float(current_cal[i]), 4)
            for i, name in enumerate(CHEST_FEATURE_NAMES)
        },
    })


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)