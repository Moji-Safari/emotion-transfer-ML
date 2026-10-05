import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.models.random_forest_model import create_random_forest


def run_loso(X, y, groups):
    """
    Perform Leave-One-Subject-Out evaluation.

    For each subject:
        - Train on all other subjects.
        - Test only on the held-out subject.

    Returns
    -------
    results : list
        Per-subject evaluation results.

    all_y_true : numpy.ndarray
        All true labels.

    all_y_pred : numpy.ndarray
        All predictions.
    """

    subjects = np.unique(groups)

    results = []

    all_y_true = []
    all_y_pred = []

    for test_subject in subjects:

        print("\n" + "=" * 70)
        print(f"TEST SUBJECT: {test_subject}")
        print("=" * 70)

        train_mask = groups != test_subject
        test_mask = groups == test_subject

        X_train = X[train_mask]
        y_train = y[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        print(f"Training samples: {len(y_train)}")
        print(f"Testing samples:  {len(y_test)}")

        model = create_random_forest()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        )

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        print("\nConfusion Matrix:")
        print(matrix)

        results.append({
            "subject": test_subject,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    return (
        results,
        np.asarray(all_y_true),
        np.asarray(all_y_pred),
    )