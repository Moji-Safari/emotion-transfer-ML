"""
TEMP feature ablation.

Question: does the +0.040 F1 gain from adding TEMP come from all 3
TEMP features, or would 1 suffice?

If `eda + temp_mean` matches `eda + all 3 TEMP`, we can ship a
4-feature wearable model instead of 6. That's a significant win for
a battery-powered device.

Part 1: LOSO with 5 feature configurations.
Part 2: per-subject ΔF1 for the two most interesting comparisons,
        to check whether the gain is outlier-driven.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, accuracy_score

from src.preprocessing.wesad_preprocessing import (
    load_subject,
    create_eda_temp_windows,
)
from src.features.eda_features import extract_eda_features
from src.features.temp_features import extract_temp_features
from src.evaluation.loso import run_loso


SUBJECTS = [
    "S2","S3","S4","S5","S6","S7","S8","S9","S10","S11",
    "S13","S14","S15","S16","S17",
]

MODEL_NAME = "svm_rbf"

EDA_KEEP = ["mean", "std", "mean_absolute_change"]
TEMP_KEEP_ALL = ["temp_mean", "temp_std", "temp_slope"]


# -------------------------------------------------------------------
# Feature matrix construction — one function per config
# -------------------------------------------------------------------

def _extract_windows_for_subject(sid):
    data = load_subject(sid)
    eda_w, temp_w, y = create_eda_temp_windows(data)
    return eda_w, temp_w, y


def build_matrix(eda_w, temp_w, config):
    """Return the feature matrix for one subject, given config name."""
    rows = []
    for e, t in zip(eda_w, temp_w):
        ef = extract_eda_features(e, sampling_rate=4)
        tf = extract_temp_features(t, sampling_rate=4)

        row = [ef[k] for k in EDA_KEEP]  # 3 EDA features

        if config == "eda_only":
            pass
        elif config == "eda_temp_mean":
            row.append(tf["temp_mean"])
        elif config == "eda_temp_std":
            row.append(tf["temp_std"])
        elif config == "eda_temp_slope":
            row.append(tf["temp_slope"])
        elif config == "eda_temp_all":
            row += [tf["temp_mean"], tf["temp_std"], tf["temp_slope"]]
        else:
            raise ValueError(f"Unknown config: {config}")

        rows.append(row)

    return np.asarray(rows, dtype=float)


def build_dataset(config):
    all_X, all_y, all_g = [], [], []
    for sid in SUBJECTS:
        eda_w, temp_w, y = _extract_windows_for_subject(sid)
        X = build_matrix(eda_w, temp_w, config)
        all_X.append(X)
        all_y.append(y)
        all_g.append(np.full(len(y), sid, dtype=object))
    return np.vstack(all_X), np.concatenate(all_y), np.concatenate(all_g)


# -------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------

def run_config(config, X, y, groups):
    print(f"\n{'-' * 78}")
    print(f"CONFIG: {config}  (X.shape = {X.shape})")
    print(f"{'-' * 78}")

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
    print("-" * 60)
    print(f"{'Subject':<10}{'Base F1':<12}{'Variant F1':<12}{'Δ':<10}")
    for i, sid in enumerate(base["subjects"]):
        b = base["per_subj_f1"][i]
        v = variant["per_subj_f1"][i]
        d = v - b
        print(f"{sid:<10}{b:<12.4f}{v:<12.4f}{d:+.4f}")


def main():
    print("=" * 78)
    print("TEMP FEATURE ABLATION — RBF SVM, LOSO, 15 subjects")
    print("=" * 78)

    configs = [
        "eda_only",
        "eda_temp_mean",
        "eda_temp_std",
        "eda_temp_slope",
        "eda_temp_all",
    ]

    summaries = {}
    for cfg in configs:
        X, y, groups = build_dataset(cfg)
        summaries[cfg] = run_config(cfg, X, y, groups)

    # ---------- Paired tests ----------
    print("\n" + "=" * 78)
    print("PAIRED TESTS (Wilcoxon, per-subject F1, n=15)")
    print("=" * 78)

    eda = summaries["eda_only"]["per_subj_f1"]

    for cfg in ["eda_temp_mean", "eda_temp_std",
                "eda_temp_slope", "eda_temp_all"]:
        var = summaries[cfg]["per_subj_f1"]
        diff = var - eda
        try:
            stat, p = wilcoxon(var, eda, zero_method="wilcox",
                               alternative="two-sided")
        except ValueError:
            p = float("nan")
        n_better = int(np.sum(diff > 0))
        n_worse = int(np.sum(diff < 0))
        print(f"\n{cfg} − eda_only:")
        print(f"  improved: {n_better:>2}/15")
        print(f"  worse:    {n_worse:>2}/15")
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

    # ---------- Per-subject delta tables ----------
    print("\n" + "=" * 78)
    print("PER-SUBJECT Δ TABLES")
    print("=" * 78)

    print_per_subject_delta(
        summaries["eda_only"],
        summaries["eda_temp_all"],
        "ΔF1: eda_temp_all − eda_only",
    )
    print_per_subject_delta(
        summaries["eda_only"],
        summaries["eda_temp_mean"],
        "ΔF1: eda_temp_mean − eda_only",
    )


if __name__ == "__main__":
    main()