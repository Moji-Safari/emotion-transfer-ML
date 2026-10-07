"""
Run LOSO for every model in MODEL_REGISTRY.

Outputs a comparison table with both:
  - aggregate metrics (window-weighted)
  - per-subject metrics (subject-weighted, mean ± std)

Reference for the protocol:
  Schmidt et al. (2018), "Introducing WESAD" — ICMI 2018
  https://doi.org/10.1145/3242969.3242985
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.data.build_dataset import build_all_subjects_dataset
from src.evaluation.loso import run_loso
from src.models.model_factory import MODEL_REGISTRY


def summarize(results, y_true, y_pred):
    f1s = np.array([r["f1"] for r in results])
    accs = np.array([r["accuracy"] for r in results])
    precs = np.array([r["precision"] for r in results])
    recs = np.array([r["recall"] for r in results])

    return {
        "acc_agg":  accuracy_score(y_true, y_pred),
        "prec_agg": precision_score(y_true, y_pred, zero_division=0),
        "rec_agg":  recall_score(y_true, y_pred, zero_division=0),
        "f1_agg":   f1_score(y_true, y_pred, zero_division=0),

        "f1_subj_mean":   f1s.mean(),
        "f1_subj_std":    f1s.std(),
        "acc_subj_mean":  accs.mean(),
        "acc_subj_std":   accs.std(),
        "prec_subj_mean": precs.mean(),
        "rec_subj_mean":  recs.mean(),
    }


def main():
    print("=" * 78)
    print("MODEL COMPARISON — SAME FEATURES, SAME LOSO, SAME CLASS WEIGHTING")
    print("=" * 78)
    print("Dataset: WESAD, 15 subjects")
    print("Modality: wrist EDA only (4 Hz)")
    print("Window: 30 s, non-overlapping (120 samples)")
    print("Features: 9 handcrafted (mean, std, min, max, range,")
    print("          median, mean_abs_change, std_change, slope)")
    print("=" * 78)

    X, y, groups = build_all_subjects_dataset()

    print(f"\nDataset shapes: X={X.shape}, y={y.shape}, "
          f"subjects={len(np.unique(groups))}")
    print(f"Class balance:  baseline={int(np.sum(y == 0))}, "
          f"stress={int(np.sum(y == 1))}")

    summaries = {}

    for model_name in MODEL_REGISTRY:
        print(f"\n{'-' * 78}")
        print(f"MODEL: {model_name}")
        print(f"{'-' * 78}")

        results, y_true, y_pred = run_loso(
            X, y, groups,
            model_name=model_name,
            verbose=True,
        )

        summaries[model_name] = summarize(results, y_true, y_pred)

    # ---------- Final comparison table ----------
    print("\n" + "=" * 78)
    print("FINAL COMPARISON")
    print("=" * 78)

    header = (
        f"{'Model':<16}"
        f"{'Acc(agg)':<10}"
        f"{'F1(agg)':<10}"
        f"{'F1(subj)':<16}"
        f"{'Acc(subj)':<16}"
    )
    print(header)
    print("-" * len(header))

    ordered = sorted(
        summaries.items(),
        key=lambda kv: kv[1]["f1_subj_mean"],
        reverse=True,
    )

    for name, s in ordered:
        print(
            f"{name:<16}"
            f"{s['acc_agg']:<10.4f}"
            f"{s['f1_agg']:<10.4f}"
            f"{s['f1_subj_mean']:<7.4f}±"
            f"{s['f1_subj_std']:<8.4f}"
            f"{s['acc_subj_mean']:<7.4f}±"
            f"{s['acc_subj_std']:<8.4f}"
        )

    # ---------- Interpretation hints ----------
    print("\n" + "=" * 78)
    print("HOW TO READ THIS TABLE")
    print("=" * 78)
    print("- 'majority' = always predicts baseline. Accuracy = baseline ratio.")
    print("  Any model below 'majority' accuracy is worse than guessing.")
    print("- F1(subj) = mean F1 across 15 held-out subjects (subject-weighted).")
    print("  This is the most honest metric for a wearable that must")
    print("  work on new people.")
    print("- Large gap between linear models (logistic, svm_linear) and")
    print("  nonlinear (svm_rbf, random_forest) => nonlinear structure matters.")
    print("- Small gap => features are the bottleneck, not the model.")


if __name__ == "__main__":
    main()