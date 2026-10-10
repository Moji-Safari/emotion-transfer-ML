"""
Diagnose specific WESAD subjects to understand per-subject anomalies.

Motivation
----------
Across multiple experiments (EDA+TEMP, EDA+TEMP+BVP), four subjects
showed large swings in F1:
    S14: rescued by BVP (F1 0.09 -> 0.95)
    S2:  broken by BVP (F1 0.71 -> 0.33)
    S10: broken by BVP (F1 0.95 -> 0.61)
    S15: broken by BVP (F1 0.98 -> 0.69)

This script inspects the raw signals for these subjects plus a
control (S4, S16) to see whether there is a technical explanation.

Outputs
-------
- Console statistics per subject
- A 3-panel figure per subject: EDA, TEMP, BVP over time, with
  baseline vs stress regions shaded.

Reference
---------
Schmidt et al. (2018), "Introducing WESAD," ICMI 2018.
  https://doi.org/10.1145/3242969.3242985
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing.wesad_preprocessing import load_subject


SUBJECTS = ["S2", "S4", "S10", "S14", "S15", "S16"]

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def signal_stats(name, x):
    return {
        "name": name,
        "n":      len(x),
        "nan":    int(np.isnan(x).sum()),
        "mean":   float(np.nanmean(x)),
        "std":    float(np.nanstd(x)),
        "min":    float(np.nanmin(x)),
        "max":    float(np.nanmax(x)),
    }


def print_stats_table(stats):
    print(f"  {'signal':<8}{'n':<10}{'nan':<8}"
          f"{'mean':<12}{'std':<12}{'min':<12}{'max':<12}")
    for s in stats:
        print(f"  {s['name']:<8}{s['n']:<10}{s['nan']:<8}"
              f"{s['mean']:<12.4f}{s['std']:<12.4f}"
              f"{s['min']:<12.4f}{s['max']:<12.4f}")


def window_means(signal, labels, fs, window_samples):
    """
    For each 30-s window, return (mean_value, majority_label).

    Uses the label at the window's center time to decide class.
    """
    n_windows = len(signal) // window_samples
    means = []
    labels_out = []

    # label is at 700 Hz, signal at fs. Label index for window center:
    label_rate = 700
    for w in range(n_windows):
        s = w * window_samples
        e = s + window_samples
        seg = signal[s:e]
        if np.isnan(seg).any():
            continue
        means.append(float(np.mean(seg)))

        center_time_sec = (s + window_samples / 2) / fs
        label_idx = int(center_time_sec * label_rate)
        if label_idx < len(labels):
            labels_out.append(int(labels[label_idx]))
        else:
            labels_out.append(-1)

    return np.array(means), np.array(labels_out)


def diagnose_subject(sid):
    print("\n" + "=" * 78)
    print(f"SUBJECT: {sid}")
    print("=" * 78)

    data = load_subject(sid)

    eda_w = data["signal"]["wrist"]["EDA"].reshape(-1)
    temp_w = data["signal"]["wrist"]["TEMP"].reshape(-1)
    bvp_w = data["signal"]["wrist"]["BVP"].reshape(-1)
    labels = data["label"]

    # ---- Basic stats ----
    print("\nWrist signal statistics:")
    print_stats_table([
        signal_stats("EDA", eda_w),
        signal_stats("TEMP", temp_w),
        signal_stats("BVP", bvp_w),
    ])

    # ---- Label distribution ----
    unique, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution (full recording):")
    for u, c in zip(unique, counts):
        pct = 100 * c / len(labels)
        print(f"  label {u}: {c:>9} ({pct:5.2f}%)")

    # ---- Window-level statistics ----
    print("\nWindow-level EDA/TEMP/BVP means by class:")

    eda_means, eda_lbls = window_means(eda_w, labels, 4, 120)
    temp_means, _      = window_means(temp_w, labels, 4, 120)
    bvp_means, _       = window_means(bvp_w, labels, 64, 1920)

    # binary label: 1 -> baseline(0), 2 -> stress(1)
    def binary(l):
        if l == 1: return 0
        if l == 2: return 1
        return -1

    eda_bin = np.array([binary(l) for l in eda_lbls])

    for name, means in [("EDA", eda_means),
                        ("TEMP", temp_means),
                        ("BVP", bvp_means)]:
        m = means[:len(eda_bin)]
        base = m[eda_bin == 0]
        stress = m[eda_bin == 1]
        if len(base) == 0 or len(stress) == 0:
            print(f"  {name}: insufficient class data")
            continue
        print(f"  {name:<6} "
              f"baseline mean = {base.mean():>10.4f}  "
              f"stress mean = {stress.mean():>10.4f}  "
              f"Δ = {stress.mean() - base.mean():>+10.4f}  "
              f"({len(base)} base / {len(stress)} stress windows)")

    # ---- Plot ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)

    # Time axes in minutes
    t_eda = np.arange(len(eda_w)) / 4 / 60
    t_temp = np.arange(len(temp_w)) / 4 / 60
    t_bvp = np.arange(len(bvp_w)) / 64 / 60

    axes[0].plot(t_eda, eda_w, lw=0.5, color="tab:blue")
    axes[0].set_ylabel("EDA (µS)")
    axes[0].set_title(f"{sid} — raw wrist signals over the recording")

    axes[1].plot(t_temp, temp_w, lw=0.5, color="tab:red")
    axes[1].set_ylabel("TEMP (°C)")

    axes[2].plot(t_bvp, bvp_w, lw=0.3, color="tab:green")
    axes[2].set_ylabel("BVP (raw)")
    axes[2].set_xlabel("Time (minutes)")

    # Shade baseline (label 1) and stress (label 2) regions
    t_labels = np.arange(len(labels)) / 700 / 60
    base_mask = (labels == 1)
    stress_mask = (labels == 2)

    for ax in axes:
        ylim = ax.get_ylim()
        ax.fill_between(
            t_labels, ylim[0], ylim[1], where=base_mask,
            color="lightgreen", alpha=0.15, label="Baseline (label 1)",
        )
        ax.fill_between(
            t_labels, ylim[0], ylim[1], where=stress_mask,
            color="salmon", alpha=0.15, label="Stress (label 2)",
        )
        ax.set_ylim(ylim)

    axes[0].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path = FIG_DIR / f"diagnose_{sid}.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"\n  Saved plot: {out_path}")


def main():
    print("=" * 78)
    print("SUBJECT DIAGNOSIS")
    print("=" * 78)
    print(f"Subjects: {SUBJECTS}")
    print(f"Figures directory: {FIG_DIR.resolve()}")

    for sid in SUBJECTS:
        diagnose_subject(sid)

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"\nOpen the PNG files in {FIG_DIR} and look at each subject's")
    print("EDA, TEMP, and BVP signal. The shaded regions mark where the")
    print("baseline and stress conditions occurred in the experiment.")


if __name__ == "__main__":
    main()