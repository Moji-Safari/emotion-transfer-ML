"""
Flask API for the calibrated stress classifier and haptic generator.

Endpoints
---------
GET  /health
    Returns {"status": "ok"}.

POST /predict
    Body:
        {
            "features": [f1, f2, f3, f4],   # raw features
            "calibration": {                # optional, per session
                "B_s": [b1, b2, b3, b4],
                "sigma_s": [s1, s2, s3, s4]
            }
        }
    Returns:
        {
            "stress_probability": float,
            "prediction": 0 | 1,
            "band": "low" | "medium" | "high"
        }

POST /calibrate
    Body:
        {
            "baseline_features": [[...], [...], ...]
        }
    Returns:
        {
            "B_s": [b1, b2, b3, b4],
            "sigma_s": [s1, s2, s3, s4]
        }
    The client stores these and includes them in subsequent
    /predict calls.

POST /generate-haptic
    Body: { "stress_probability": float }
    Returns: the haptic pattern from stress_to_haptic().

Run:
    python -m api.app
    or
    flask --app api.app run --port 5000
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
CORS(app)  # allow React dev server to call from a different port


# ------------------------------------------------------------------
# Load model at startup
# ------------------------------------------------------------------
def _load_model():
    svm_path = MODEL_DIR / "svm.pkl"
    names_path = MODEL_DIR / "feature_names.json"
    if not svm_path.exists():
        raise FileNotFoundError(
            f"Model not found at {svm_path}. "
            f"Run scripts.export_model first."
        )
    with open(svm_path, "rb") as f:
        model = pickle.load(f)
    with open(names_path) as f:
        feature_names = json.load(f)
    return model, feature_names


model, FEATURE_NAMES = _load_model()
print(f"Loaded model. Expects features: {FEATURE_NAMES}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _calibrate(X: np.ndarray, B_s: np.ndarray,
               sigma_s: np.ndarray) -> np.ndarray:
    """Apply (X - B_s) / max(sigma_s, floor)."""
    denom = np.maximum(sigma_s, FLOOR)
    return (X - B_s) / denom


def _validate_features(features) -> np.ndarray:
    arr = np.asarray(features, dtype=float)
    if arr.shape != (4,):
        raise ValueError(
            f"Expected 4 features, got shape {arr.shape}"
        )
    return arr.reshape(1, -1)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_dir": str(MODEL_DIR.resolve()),
        "features": FEATURE_NAMES,
    })


@app.route("/calibrate", methods=["POST"])
def calibrate():
    """Compute calibration statistics from baseline windows."""
    data = request.get_json(silent=True) or {}
    baseline = data.get("baseline_features")

    if baseline is None:
        return jsonify({"error": "missing 'baseline_features'"}), 400

    arr = np.asarray(baseline, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        return jsonify({
            "error": f"baseline_features must be (n, 4), "
                     f"got {arr.shape}"
        }), 400
    if arr.shape[0] < 2:
        return jsonify({
            "error": "need at least 2 baseline windows"
        }), 400

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

    # Extract raw features
    try:
        X_raw = _validate_features(data.get("features"))
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    # Optional calibration
    cal = data.get("calibration")
    if cal is not None:
        try:
            B_s = np.asarray(cal["B_s"], dtype=float)
            sigma_s = np.asarray(cal["sigma_s"], dtype=float)
            if B_s.shape != (4,) or sigma_s.shape != (4,):
                raise ValueError("B_s and sigma_s must have length 4")
            X = _calibrate(X_raw, B_s, sigma_s)
        except (KeyError, ValueError, TypeError) as e:
            return jsonify({
                "error": f"invalid calibration: {e}"
            }), 400
    else:
        X = X_raw  # uncalibrated — will be less accurate

    # Predict
    # The SVM pipeline returns 0 or 1 from .predict, but we want a
    # probability. sklearn's SVC has `probability=False` in our
    # model factory, so we use decision_function and squash it.
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0, 1])
    else:
        # decision_function → sigmoid to get a pseudo-probability
        score = float(model.decision_function(X)[0])
        proba = 1.0 / (1.0 + np.exp(-score))

    pred = int(proba >= 0.5)

    if proba < 0.35:
        band = "low"
    elif proba < 0.65:
        band = "medium"
    else:
        band = "high"

    return jsonify({
        "stress_probability": round(proba, 4),
        "prediction": pred,
        "band": band,
        "calibrated": cal is not None,
    })


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


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)