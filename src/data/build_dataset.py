import numpy as np

from src.preprocessing.wesad_preprocessing import (
    load_subject,
    create_eda_windows,
)

from src.features.build_features import build_feature_matrix


SUBJECTS = [
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
]


def build_subject_dataset(subject_id):
    """
    Load one WESAD subject and convert their EDA signal
    into a feature matrix and corresponding labels.
    """

    data = load_subject(subject_id)

    X_windows, y = create_eda_windows(data)

    X_features = build_feature_matrix(X_windows)

    return X_features, y


def build_all_subjects_dataset():
    """
    Build the complete dataset from all available WESAD subjects.

    Returns
    -------
    X : numpy.ndarray
        Feature matrix.

    y : numpy.ndarray
        Labels.

    groups : numpy.ndarray
        Subject ID for every sample.
    """

    all_X = []
    all_y = []
    all_groups = []

    for subject_id in SUBJECTS:
        print(f"Processing {subject_id}...")

        X, y = build_subject_dataset(subject_id)

        print(f"  Windows: {len(y)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Baseline: {np.sum(y == 0)}")
        print(f"  Stress:   {np.sum(y == 1)}")

        all_X.append(X)
        all_y.append(y)

        subject_groups = np.full(
            len(y),
            subject_id,
            dtype=object,
        )

        all_groups.append(subject_groups)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    groups = np.concatenate(all_groups)

    return X, y, groups