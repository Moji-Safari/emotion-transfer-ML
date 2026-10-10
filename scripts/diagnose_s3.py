"""
Diagnose S3's collapse under std-based calibration.

Background
----------
S3's F1 went from 0.829 (mean calibration) to 0.174 (std
calibration), a −0.655 drop. This is the largest single-subject
swing in the calibration experiments. We need to know whether
this is due to a numerical issue (near-zero sigma) or a genuine
modelling failure.

The script prints, for S3 and three control subjects (S4, S14,
S16):
  - Baseline window count
  - Per-feature baseline mean and std (B_s and sigma_s)
  - The calibration amplification factor (1/sigma_s)
  - Calibrated stress feature statistics
  - Comparison to the training distribution (out-of-distribution
    check)

Interpretation
--------------
If any sigma_s < 1e-3, the division X* = (X - B_s) / sigma_s
produces huge values and the RBF kernel sees off-scale features.
That would explain S3's collapse and would be fixable.

If sigma_s values are all normal (>0.01), then S3's collapse has
a different cause: likely the calibrated feature vectors genuinely
sit outside the training distribution.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.preprocessing.wesad_preprocessing import (
    load_subject, create_all_windows,
)
from src.features.eda_features import extract_eda_features
from src.features.temp_features import extract_temp_features


FEATURE_NAMES = [
    "eda_mean",
    "eda_std",
    "eda_mean_abs_change",
    "temp_mean",
]

DIAGNOSE = ["S3", "S4", "S14", "S16"]  # S3 first, then controls


def build_subject_features(sid):
    """Return X_subject, y_subject using the 4-feature set."""
    data = load_subject(sid)
    eda_w, temp_w, _, y = create_all_windows(data)

    rows = []
    for e, t in zip(eda_w, temp_w):
        ef = extract_eda_features(e, sampling_rate=4)
        tf = extract_temp_features(t, sampling_rate=4)
        rows.append([
            ef["mean"],
            ef["std"],
            ef["mean_absolute_change"],
            tf["temp_mean"],
        ])
    return np.asarray(rows, dtype=float), y


def diagnose(sid, reference_stats=None):
    print("\n" + "=" * 78)
    print(f"SUBJECT: {sid}")
    print("=" * 78)

    X, y = build_subject_features(sid)
    print(f"Total windows: {len(y)}")
    print(f"  baseline: {int((y == 0).sum())}")
    print(f"  stress:   {int((y == 1).sum())}")

    baseline_mask = (y == 0)
    stress_mask = (y == 1)

    if baseline_mask.sum() < 2:
        print("  ERROR: fewer than 2 baseline windows; cannot calibrate.")
        return None

    B_s = X[baseline_mask].mean(axis=0)
    sigma_s = X[baseline_mask].std(axis=0)

    print("\nBaseline statistics (per feature):")
    print(f"  {'feature':<24}{'mean (B_s)':<16}{'std (sigma_s)':<18}"
          f"{'1/sigma_s':<14}")
    print("  " + "-" * 72)

    for i, name in enumerate(FEATURE_NAMES):
        amp = 1.0 / sigma_s[i] if sigma_s[i] > 0 else float("inf")
        flag = " <-- tiny!" if sigma_s[i] < 1e-3 else ""
        print(f"  {name:<24}{B_s[i]:<16.6f}{sigma_s[i]:<18.6f}"
              f"{amp:<14.2f}{flag}")

    # ------------------------------------------------------------------
    # Calibrated feature statistics
    # ------------------------------------------------------------------
    print("\nCalibrated stress feature statistics "
          "(X* = (X - B_s) / sigma_s):")

    X_stress = X[stress_mask]
    X_cal = (X_stress - B_s) / (sigma_s + 1e-6)

    print(f"  {'feature':<24}{'mean':<14}{'std':<14}"
          f"{'min':<14}{'max':<14}")
    print("  " + "-" * 74)

    for i, name in enumerate(FEATURE_NAMES):
        col = X_cal[:, i]
        print(f"  {name:<24}{col.mean():<14.4f}{col.std():<14.4f}"
              f"{col.min():<14.4f}{col.max():<14.4f}")

    # ------------------------------------------------------------------
    # How extreme is this relative to the training distribution?
    # ------------------------------------------------------------------
    # We can't compute the training distribution here (needs other
    # subjects), but we can check whether any calibrated stress
    # feature exceeds a reasonable range (say |value| > 10).
    print("\nExtremeness check (values outside [-10, 10] are suspicious):")
    for i, name in enumerate(FEATURE_NAMES):
        col = X_cal[:, i]
        n_extreme = int((np.abs(col) > 10).sum())
        pct = 100.0 * n_extreme / len(col) if len(col) else 0
        flag = " <-- EXTREME" if n_extreme > 0 else ""
        print(f"  {name:<24}{n_extreme:>4} / {len(col):<4} "
              f"({pct:5.1f}%){flag}")

    return {
        "subject": sid,
        "B_s": B_s,
        "sigma_s": sigma_s,
        "X_cal_stress": X_cal,
    }


def main():
    print("=" * 78)
    print("S3 DIAGNOSTIC — investigating the std-calibration collapse")
    print("=" * 78)

    results = {}
    for sid in DIAGNOSE:
        results[sid] = diagnose(sid)

    # ------------------------------------------------------------------
    # Cross-subject comparison of sigma_s
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("CROSS-SUBJECT SUMMARY: sigma_s values per feature")
    print("=" * 78)
    print(f"  {'subject':<10}"
          + "".join(f"{n:<18}" for n in FEATURE_NAMES))
    print("  " + "-" * 78)

    for sid, r in results.items():
        if r is None:
            continue
        row = f"  {sid:<10}"
        for s in r["sigma_s"]:
            row += f"{s:<18.6f}"
        print(row)

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("INTERPRETATION GUIDE")
    print("=" * 78)
    print("Look at S3's sigma_s row:")
    print("  - If any sigma_s is very small (< 0.001), the division")
    print("    amplifies that feature enormously. The RBF SVM sees")
    print("    values far outside its training range. This is a")
    print("    numerical problem — fixable by adding a floor on")
    print("    sigma_s.")
    print("  - If all sigma_s are normal (> 0.01) for S3, but the")
    print("    calibrated stress values are still extreme, S3's")
    print("    baseline is genuinely unrepresentative of its stress")
    print("    period. That is a modelling failure, not a numerical")
    print("    one.")
    print()
    print("Compare S3's sigma_s row to S4's (a control that works).")
    print("If S3's values are much smaller or larger, that is the")
    print("explanation.")


if __name__ == "__main__":
    main()