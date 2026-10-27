"""
Diagnose why the chest 4-class model predicts only one class.

Prints:
  - Feature scaler mean and scale (if Pipeline)
  - Decision function or probability distribution for sample inputs
  - The class distribution during training
  - The actual feature values produced by the heuristic
"""

import sys
import pickle
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


MODEL_DIR = Path("models/deployed")


def load_model():
    with open(MODEL_DIR / "chest_svm.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "chest_feature_names.json") as f:
        names = json.load(f)
    return model, names


def inspect_model(model, names):
    print("=" * 70)
    print("MODEL INSPECTION")
    print("=" * 70)
    print(f"Model type: {type(model).__name__}")

    # Check for Pipeline
    if hasattr(model, "named_steps"):
        print(f"Pipeline steps: {list(model.named_steps.keys())}")
        for step_name, step in model.named_steps.items():
            print(f"  {step_name}: {type(step).__name__}")

        if "scaler" in model.named_steps:
            scaler = model.named_steps["scaler"]
            if hasattr(scaler, "mean_"):
                print("\nScaler mean (training feature mean):")
                for i, n in enumerate(names):
                    print(f"  {n:<25} {scaler.mean_[i]:>12.4f}")
                print("\nScaler scale (training feature std):")
                for i, n in enumerate(names):
                    print(f"  {n:<25} {scaler.scale_[i]:>12.4f}")

        if "clf" in model.named_steps:
            clf = model.named_steps["clf"]
            print(f"\nClassifier: {type(clf).__name__}")
            if hasattr(clf, "classes_"):
                print(f"Classes: {clf.classes_}")
            if hasattr(clf, "support_vectors_"):
                print(f"Number of support vectors: {len(clf.support_vectors_)}")
    else:
        print("Model is not a Pipeline")
        if hasattr(model, "classes_"):
            print(f"Classes: {model.classes_}")


def test_input(model, names, label, values):
    print(f"\n{'=' * 70}")
    print(f"TEST: {label}")
    print(f"{'=' * 70}")
    print("Input features:")
    for n, v in zip(names, values):
        print(f"  {n:<25} {v:>12.4f}")

    X = np.array([values], dtype=float)

    # Check what the scaler does
    if hasattr(model, "named_steps") and "scaler" in model.named_steps:
        scaler = model.named_steps["scaler"]
        X_scaled = scaler.transform(X)
        print("\nScaled features (z-score from training mean):")
        for i, n in enumerate(names):
            print(f"  {n:<25} {X_scaled[0][i]:>+12.2f}")

    # Predict
    try:
        proba = model.predict_proba(X)[0]
        print(f"\nProbabilities: {proba}")
        print(f"Predicted class: {model.predict(X)[0]}")
    except Exception as e:
        print(f"\npredict_proba failed: {e}")

    try:
        score = model.decision_function(X)
        print(f"Decision function: {score}")
    except Exception as e:
        print(f"decision_function failed: {e}")


def main():
    model, names = load_model()
    inspect_model(model, names)

    # The heuristic from the current API
    samples = {
        "amusement": {
            "hr": 84, "rmssd": 38, "sdnn": 58,
            "rr": 18, "re": 0.045, "rv": 0.65, "temp": 33.2,
        },
        "baseline": {
            "hr": 68, "rmssd": 55, "sdnn": 82,
            "rr": 14, "re": 0.012, "rv": 0.35, "temp": 33.0,
        },
        "stress": {
            "hr": 96, "rmssd": 22, "sdnn": 35,
            "rr": 22, "re": 0.085, "rv": 0.95, "temp": 31.8,
        },
        "meditation": {
            "hr": 62, "rmssd": 68, "sdnn": 105,
            "rr": 10, "re": 0.008, "rv": 0.25, "temp": 34.2,
        },
    }

    for label, s in samples.items():
        lfhf = 0.3 + (s["hr"] - 60.0) / 10.0 + (60.0 - s["rmssd"]) / 20.0
        lfhf = max(0.3, min(8.0, lfhf))
        tp = max(20000.0, min(150000.0, s["rmssd"] * 1500.0))
        autocorr = max(300.0, min(800.0, 300.0 + s["rr"] * 20.0))

        values = [
            s["hr"], s["rmssd"], s["sdnn"], lfhf, tp,
            s["rr"], s["re"], autocorr, s["rv"],
        ]
        test_input(model, names, label, values)


if __name__ == "__main__":
    main()