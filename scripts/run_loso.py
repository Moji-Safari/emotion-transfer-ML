import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.data.build_dataset import build_all_subjects_dataset
from src.evaluation.loso import run_loso


def main():

    print("=" * 70)
    print("WESAD LEAVE-ONE-SUBJECT-OUT EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Build complete dataset
    # ---------------------------------------------------------

    X, y, groups = build_all_subjects_dataset()

    print("\nComplete dataset:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")

    # ---------------------------------------------------------
    # Run LOSO
    # ---------------------------------------------------------

    results, y_true, y_pred = run_loso(
        X,
        y,
        groups,
    )

    # ---------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    # ---------------------------------------------------------
    # Print final results
    # ---------------------------------------------------------

    print("\n\n" + "=" * 70)
    print("LOSO FINAL RESULTS")
    print("=" * 70)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nOverall Confusion Matrix:")
    print(matrix)

    # ---------------------------------------------------------
    # Per-subject summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-SUBJECT SUMMARY")
    print("=" * 70)

    print(
        f"{'Subject':<10}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
    )

    for result in results:

        print(
            f"{result['subject']:<10}"
            f"{result['accuracy']:<12.4f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1']:<12.4f}"
        )

    # ---------------------------------------------------------
    # Mean and standard deviation across subjects
    # ---------------------------------------------------------

    accuracies = np.array(
        [result["accuracy"] for result in results]
    )

    precisions = np.array(
        [result["precision"] for result in results]
    )

    recalls = np.array(
        [result["recall"] for result in results]
    )

    f1_scores = np.array(
        [result["f1"] for result in results]
    )

    print("\n" + "=" * 70)
    print("MEAN ± STANDARD DEVIATION")
    print("=" * 70)

    print(
        f"Accuracy:  {accuracies.mean():.4f} ± "
        f"{accuracies.std():.4f}"
    )

    print(
        f"Precision: {precisions.mean():.4f} ± "
        f"{precisions.std():.4f}"
    )

    print(
        f"Recall:    {recalls.mean():.4f} ± "
        f"{recalls.std():.4f}"
    )

    print(
        f"F1 Score:  {f1_scores.mean():.4f} ± "
        f"{f1_scores.std():.4f}"
    )


if __name__ == "__main__":
    main()