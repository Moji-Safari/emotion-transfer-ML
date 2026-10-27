"""
Export the four-class chest model and its calibration statistics.

Trains on all 15 subjects using 60s chest windows with calibration.
Saves the model, the calibration statistics per subject, and the
feature names.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pickle

from src.preprocessing.wesad_preprocessing import load_subject, create_chest_windows
from src.features.build_chest_features import build_chest_feature_matrix
from src.preprocessing.calibration import CalibrationTransformerStdFloored
from src.models.model_factory import make_svm_rbf


SUBJECTS = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11",
            "S13","S14","S15","S16","S17"]
FLOOR = 0.01

OUT_DIR = Path("models/deployed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Building chest dataset...")

    all_X, all_y = [], []
    feature_names = None

    for sid in SUBJECTS:
        data = load_subject(sid)
        ecg_w, resp_w, y = create_chest_windows(data)
        if len(y) == 0:
            continue

        X, names = build_chest_feature_matrix(ecg_w, resp_w)
        if feature_names is None:
            feature_names = names

        # Per-subject calibration
        ct = CalibrationTransformerStdFloored(floor=FLOOR)
        baseline_mask = (y == 0)
        if baseline_mask.sum() < 2:
            continue
        ct.fit(X[baseline_mask])
        X = ct.transform(X)

        # Drop NaN
        valid = ~np.isnan(X).any(axis=1)
        X, y = X[valid], y[valid]

        all_X.append(X)
        all_y.append(y)
        print(f"  {sid}: {len(y)} windows, classes={np.bincount(y, minlength=4)}")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    print(f"\nTotal: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y, minlength=4)}")
    print(f"Features: {feature_names}")

    print("\nTraining four-class RBF SVM...")
    model = make_svm_rbf()
    model.fit(X, y)
    print("  Done.")

    with open(OUT_DIR / "chest_svm.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(OUT_DIR / "chest_feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(OUT_DIR / "chest_metadata.json", "w") as f:
        json.dump({
            "model": "RBF SVM (4-class)",
            "classes": ["baseline", "stress", "amusement", "meditation"],
            "features": feature_names,
            "n_windows": int(len(y)),
            "class_counts": np.bincount(y, minlength=4).tolist(),
            "calibration": {"method": "std_floored", "floor": FLOOR},
        }, f, indent=2)

    print(f"\nSaved to {OUT_DIR}/chest_svm.pkl")


if __name__ == "__main__":
    main()