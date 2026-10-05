from pathlib import Path
import pickle

import numpy as np


# ============================================================
# Configuration
# ============================================================

WESAD_ROOT = Path(r"E:\dataset\WESAD\WESAD")

# WESAD chest/label sampling rate
CHEST_FS = 700

# Wrist sampling rates
WRIST_EDA_FS = 4
WRIST_TEMP_FS = 4
WRIST_BVP_FS = 64

# Window configuration
WINDOW_SECONDS = 30

# We only keep these two WESAD conditions
BASELINE_LABEL = 1
STRESS_LABEL = 2


# ============================================================
# Loading
# ============================================================

def load_subject(subject_id):
    """
    Load one subject's synchronized WESAD data.

    Parameters
    ----------
    subject_id : str
        Example: "S2"

    Returns
    -------
    dict
        The contents of the subject's .pkl file.
    """

    subject_path = WESAD_ROOT / subject_id / f"{subject_id}.pkl"

    if not subject_path.exists():
        raise FileNotFoundError(
            f"Could not find WESAD file:\n{subject_path}"
        )

    with open(subject_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    return data


# ============================================================
# Label handling
# ============================================================

def get_baseline_stress_labels(labels):
    """
    Convert the original WESAD labels into binary labels.

    Original WESAD:
        1 = baseline
        2 = stress

    New labels:
        0 = baseline
        1 = stress

    All other labels are marked as -1.
    """

    binary_labels = np.full(labels.shape, -1, dtype=np.int8)

    binary_labels[labels == BASELINE_LABEL] = 0
    binary_labels[labels == STRESS_LABEL] = 1

    return binary_labels


# ============================================================
# Window label
# ============================================================

def get_window_label(labels):
    """
    Determine the label of one window.

    A window is valid only if every sample has the
    same valid label.

    Returns
    -------
    int
        0 = baseline
        1 = stress
        -1 = invalid/mixed/ignored
    """

    unique_labels = np.unique(labels)

    if len(unique_labels) != 1:
        return -1

    label = unique_labels[0]

    if label not in (0, 1):
        return -1

    return int(label)


# ============================================================
# Wrist EDA windows
# ============================================================

def create_eda_windows(data):
    """
    Create 30-second windows from wrist EDA.

    EDA is sampled at 4 Hz.
    Therefore:

        30 seconds × 4 samples/sec = 120 samples/window

    Labels are sampled at 700 Hz, so we use the corresponding
    700 Hz label samples for the same time interval.
    """

    wrist_eda = data["signal"]["wrist"]["EDA"].reshape(-1)
    labels = data["label"]

    samples_per_window = WINDOW_SECONDS * WRIST_EDA_FS

    # Number of EDA samples corresponding to one EDA sample
    label_samples_per_eda_sample = CHEST_FS // WRIST_EDA_FS

    # Convert the label sequence to the 4 Hz timeline.
    #
    # Every 175 chest samples correspond to approximately
    # one wrist EDA sample.
    label_indices = (
        np.arange(len(wrist_eda))   
        * label_samples_per_eda_sample
    )   
        
    # Do not go beyond the available label array.
    valid = label_indices < len(labels)

    wrist_eda = wrist_eda[valid]
    label_indices = label_indices[valid]

    eda_labels = labels[label_indices]

    binary_labels = get_baseline_stress_labels(eda_labels)

    X = []
    y = []

    number_of_windows = len(wrist_eda) // samples_per_window

    for window_index in range(number_of_windows):

        start = window_index * samples_per_window
        end = start + samples_per_window

        eda_window = wrist_eda[start:end]
        label_window = binary_labels[start:end]

        window_label = get_window_label(label_window)

        if window_label == -1:
            continue

        X.append(eda_window)
        y.append(window_label)

    return np.asarray(X), np.asarray(y)


# ============================================================
# Main test
# ============================================================

def main():

    print("=" * 70)
    print("WESAD PREPROCESSING TEST")
    print("=" * 70)

    print(f"WESAD path: {WESAD_ROOT}")
    print(f"Window size: {WINDOW_SECONDS} seconds")
    print()

    data = load_subject("S2")

    print("Loaded subject:", data["subject"])

    print("\nAvailable wrist signals:")

    for name, values in data["signal"]["wrist"].items():
        print(f"  {name}: {values.shape}")

    print("\nCreating EDA windows...")

    X, y = create_eda_windows(data)

    print("\nResult")
    print("-" * 70)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if len(y) > 0:
        print("\nClass distribution:")

        unique, counts = np.unique(y, return_counts=True)

        for label, count in zip(unique, counts):
            name = "baseline" if label == 0 else "stress"
            print(f"  {label} ({name}): {count}")

        print("\nFirst window:")
        print(X[0])

        print("\nFirst window label:")
        print(y[0])


if __name__ == "__main__":
    main()