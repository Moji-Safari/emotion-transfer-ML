"""
BVP feature ablation.

Motivation
----------
The EDA+TEMP+BVP model (7 features) achieved F1(subj) = 0.8247, but
the paired test showed p = 0.69 — not significant. Per-subject
analysis revealed the mean gain was driven almost entirely by one
subject (S14, +0.778), while three others (S10, S2, S15) got
substantially worse.

This script tests whether a subset of BVP features can recover the
useful contribution (S14 rescue) without breaking the subjects that
were harmed. Specifically:

    - Is only one BVP feature doing the work (like temp_mean did)?
    - Does dropping bvp_hr_std (the noisiest candidate) help?
    - Does any configuration produce a significant paired test?

Reference
---------
Bouthillier et al. (2021), "Accounting for Variance in Machine
Learning Benchmarks", MLSys.
  https://proceedings.mlsys.org/paper/2021/hash/cf004fdc76fa1a4f25f62e0eb5261ca3-Abstract.html
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, accuracy_score

from src.preprocessing.wesad_preprocessing import (
    load_subject,
    create_all_windows,
)
from src.features.eda_features import extract_eda_features
from src.features.temp_features import extract_temp_features
from src.features.bvp_features import extract_bvp_features
from src.evaluation.loso import run_loso


SUBJECTS = [
    "S2","S3","S4","S5","S6","S7","S8","S9","S10","S11",
    "S13","S14","S15","S16","S17",
]

MODEL_NAME = "svm_rbf"

EDA_KEEP = ["mean", "std", "mean_absolute_change"]


# ------------------------------------------------------------------
# Build a feature matrix for one config
# ------------------------------------------------------------------

CONFIG_FEATURES = {
    "base":                 [],
    "bvp_hr_mean":          ["bvp_hr_mean"],
    "bvp_hr_std":           ["bvp_hr_std"],
    "bvp_amp":              ["bvp_pulse_amplitude_mean"],
    "bvp_hr_mean_amp":      ["bvp_hr_mean", "bvp_pulse_amplitude_mean"],
    "bvp_all":              ["bvp_hr_mean", "bvp_hr_std",
                             "bvp_pulse_amplitude_mean"],
}


def extract_row(e, t, b, bvp_features):
    ef = extract_eda_features(e, sampling_rate=4)
    tf = extract_temp_features(t, sampling_rate=4)
    bf = extract_bvp_features(b, sampling_rate=64)

    row = [ef[k] for k in EDA_KEEP] + [tf["temp_mean"]]
    for name in bvp_features:
        row.append(bf[name])
    return row


def build_dataset(config):
    """Returns X, y, groups, and a report of dropped windows."""
    bvp_features = CONFIG_FEATURES[config]
    all_X, all_y, all_g = [], [], []
    total_dropped = 0
    total_windows = 0

    for sid in SUBJECTS:
        data = load_subject(sid)
        eda_w, temp_w, bvp_w, y = create_all_windows(data)

        rows, kept = [], []
        for e, t, b, lab in zip(eda_w, temp_w, bvp_w, y):
            total_windows += 1
            row = extract_row(e, t, b, bvp_features)
            if any(np.isnan(v) for v in row):
                total_dropped += 1
                kept.append(False)
            else:
                rows.append(row)
                kept.append(True)

        if not any(kept):
            print(f"  WARNING: {sid} has no valid windows for config {config}")
            continue

        rows = np.asarray(rows, dtype=float)
        y_kept = y[np.asarray(kept, dtype=bool)]

        all_X.append(rows)
        all_y.append(y_kept)
        all_g.append(np.full(len(y_kept), sid, dtype=object))

    if total_dropped > 0:
        print(f"  [config {config}] dropped {total_dropped}/"
              f"{total_windows} windows (NaN)")

    return np.vstack(all_X), np.concatenate(all_y), np.concatenate(all_g)


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

def run_config(config):
    print(f"\n{'-' * 78}")
    print(f"CONFIG: {config}")
    print(f"  extras: {CONFIG_FEATURES[config]}")
    print(f"{'-' * 78}")

    X, y, groups = build_dataset(config)
    print(f"  X.shape = {X.shape}")

    results, y_true, y_pred = run_loso(
        X, y, groups, model_name=MODEL_NAME, verbose=False,
    )
    f1s = np.array([r["f1"] for r in results])

    summary = {
        "config": config,
        "n_features": X.shape[1],
        "f1_subj_mean": f1s.mean(),
        "f1_subj_std":  f1s.std(),
        "f1_agg":       f1_score(y_true, y_pred, zero_division=0),
        "acc_agg":      accuracy_score(y_true, y_pred),
        "per_subj_f1":  f1s,
        "subjects":     [r["subject"] for r in results],
    }

    print(f"  F1(subj) = {summary['f1_subj_mean']:.4f} ± "
          f"{summary['f1_subj_std']:.4f}")
    print(f"  F1(agg)  = {summary['f1_agg']:.4f}")
    print(f"  Acc(agg) = {summary['acc_agg']:.4f}")

    return summary


def print_per_subject_delta(base, variant, title):
    print(f"\n{title}")
    print("-" * 62)
    print(f"{'Subject':<10}{'Base':<12}{'Variant':<12}{'Δ':<10}")
    base_by_subj = dict(zip(base["subjects"], base["per_subj_f1"]))
    var_by_subj = dict(zip(variant["subjects"], variant["per_subj_f1"]))
    for s in sorted(base_by_subj):
        b = base_by_subj[s]
        v = var_by_subj.get(s, np.nan)
        d = v - b if not np.isnan(v) else np.nan
        print(f"{s:<10}{b:<12.4f}{v:<12.4f}{d:+.4f}")


def main():
    print("=" * 78)
    print("BVP FEATURE ABLATION — RBF SVM, LOSO, 15 subjects")
    print("=" * 78)

    configs = [
        "base",
        "bvp_hr_mean",
        "bvp_hr_std",
        "bvp_amp",
        "bvp_hr_mean_amp",
        "bvp_all",
    ]

    summaries = {}
    for cfg in configs:
        summaries[cfg] = run_config(cfg)

    # ---------- Paired tests vs base ----------
    print("\n" + "=" * 78)
    print("PAIRED TESTS (Wilcoxon, per-subject F1, n=15)")
    print("=" * 78)

    base_f1_by_subj = dict(zip(
        summaries["base"]["subjects"], summaries["base"]["per_subj_f1"]
    ))

    for cfg in ["bvp_hr_mean", "bvp_hr_std", "bvp_amp",
                "bvp_hr_mean_amp", "bvp_all"]:
        var_by_subj = dict(zip(
            summaries[cfg]["subjects"], summaries[cfg]["per_subj_f1"]
        ))
        common = sorted(set(base_f1_by_subj) & set(var_by_subj))
        a = np.array([base_f1_by_subj[s] for s in common])
        b = np.array([var_by_subj[s] for s in common])
        diff = b - a

        try:
            stat, p = wilcoxon(b, a, zero_method="wilcox",
                               alternative="two-sided")
        except ValueError:
            p = float("nan")

        print(f"\n{cfg} − base:")
        print(f"  improved: {int(np.sum(diff > 0)):>2}/{len(common)}")
        print(f"  worse:    {int(np.sum(diff < 0)):>2}/{len(common)}")
        print(f"  mean Δ:   {diff.mean():+.4f}")
        print(f"  median Δ: {np.median(diff):+.4f}")
        print(f"  p = {p:.4f} "
              f"{'(significant)' if p < 0.05 else '(not significant)'}")

    # ---------- Summary table ----------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = (
        f"{'Config':<18}"
        f"{'n_feat':<8}"
        f"{'F1(subj)':<18}"
        f"{'F1(agg)':<10}"
        f"{'Acc(agg)':<10}"
    )
    print(header)
    print("-" * len(header))

    ordered = sorted(
        summaries.values(),
        key=lambda s: s["f1_subj_mean"],
        reverse=True,
    )
    for s in ordered:
        print(
            f"{s['config']:<18}"
            f"{s['n_features']:<8}"
            f"{s['f1_subj_mean']:<7.4f}±"
            f"{s['f1_subj_std']:<8.4f}"
            f"{s['f1_agg']:<10.4f}"
            f"{s['acc_agg']:<10.4f}"
        )

    # ---------- Per-subject deltas ----------
    print("\n" + "=" * 78)
    print("PER-SUBJECT Δ TABLES")
    print("=" * 78)

    for cfg in ["bvp_hr_mean", "bvp_hr_mean_amp"]:
        print_per_subject_delta(
            summaries["base"],
            summaries[cfg],
            f"ΔF1: {cfg} − base",
        )


if __name__ == "__main__":
    main()