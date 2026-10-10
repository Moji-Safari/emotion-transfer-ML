"""
Train the final deployed SVM on all 15 subjects and save it to disk.

The deployed model is trained on ALL data (not LOSO). LOSO was used
for evaluation only; for deployment we use all available data.

Saved artifacts (in models/deployed/):
    svm.pkl                 — trained sklearn Pipeline
    feature_names.json      — order of features the model expects
    metadata.json           — training info, subject list, date
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pickle

from src.preprocessing.wesad_preprocessing import (
    load_subject, create_all_windows,
)
from src.features.eda_features import extract_eda_features
from src.features.temp_features import extract_temp_features
from src.preprocessing.calibration import CalibrationTransformerStdFloored
from src.models.model_factory import make_svm_rbf


SUBJECTS = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11",
            "S13","S14","S15","S16","S17"]
EDA_KEEP = ["mean", "std", "mean_absolute_change"]
FEATURE_ORDER = EDA_KEEP + ["temp_mean"]
FLOOR = 0.01

OUT_DIR = Path("models/deployed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_calibrated_dataset():
    """
    Build the full feature matrix, calibrated per subject using
    that subject's baseline windows.

    Returns
    -------
    X_cal : (n, 4)
    y : (n,)
    subjects : (n,)  subject id per row
    """
    all_X, all_y, all_g = [], [], []

    for sid in SUBJECTS:
        data = load_subject(sid)
        eda_w, temp_w, _, y = create_all_windows(data)

        # Build 4 features per window
        rows = []
        for e, t in zip(eda_w, temp_w):
            ef = extract_eda_features(e, sampling_rate=4)
            tf = extract_temp_features(t, sampling_rate=4)
            rows.append([ef[k] for k in EDA_KEEP] + [tf["temp_mean"]])
        X = np.asarray(rows, dtype=float)

        # Calibrate per subject using their own baseline windows
        ct = CalibrationTransformerStdFloored(floor=FLOOR)
        baseline_mask = (y == 0)
        ct.fit(X[baseline_mask])
        X_cal = ct.transform(X)

        all_X.append(X_cal)
        all_y.append(y)
        all_g.append(np.full(len(y), sid, dtype=object))

    return (np.vstack(all_X),
            np.concatenate(all_y),
            np.concatenate(all_g))


def main():
    print("=" * 70)
    print("EXPORT DEPLOYED MODEL")
    print("=" * 70)

    print("\nBuilding calibrated dataset...")
    X, y, groups = build_calibrated_dataset()
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}  (baseline: {(y==0).sum()}, "
          f"stress: {(y==1).sum()})")
    print(f"  subjects: {len(np.unique(groups))}")

    print("\nTraining RBF SVM on all data...")
    model = make_svm_rbf()
    model.fit(X, y)
    print("  Done.")

    # Save artifacts
    svm_path = OUT_DIR / "svm.pkl"
    with open(svm_path, "wb") as f:
        pickle.dump(model, f)

    names_path = OUT_DIR / "feature_names.json"
    with open(names_path, "w") as f:
        json.dump(FEATURE_ORDER, f, indent=2)

    meta_path = OUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "trained_on_subjects": SUBJECTS,
            "n_windows": int(len(y)),
            "n_baseline": int((y == 0).sum()),
            "n_stress": int((y == 1).sum()),
            "features": FEATURE_ORDER,
            "calibration": {
                "method": "std_with_floor",
                "formula": "(X - B_s) / max(sigma_s, floor)",
                "floor": FLOOR,
            },
            "model": "RBF SVM (C=1.0, gamma=scale, class_weight=balanced)",
        }, f, indent=2)

    print(f"\nSaved:")
    print(f"  {svm_path}")
    print(f"  {names_path}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()