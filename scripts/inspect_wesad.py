from pathlib import Path
import pickle


WESAD_ROOT = Path(r"E:\dataset\WESAD\WESAD")


def inspect_subject(subject_dir: Path):
    subject_id = subject_dir.name
    pkl_path = subject_dir / f"{subject_id}.pkl"

    print("=" * 70)
    print(f"SUBJECT: {subject_id}")
    print("=" * 70)

    if not pkl_path.exists():
        print(f"ERROR: {pkl_path} does not exist")
        return

    with open(pkl_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    print("Top-level keys:")
    print(list(data.keys()))

    print("\nSubject:")
    print(data.get("subject"))

    print("\nSignals:")

    signals = data.get("signal", {})

    for location, location_signals in signals.items():
        print(f"\n[{location}]")

        for signal_name, values in location_signals.items():
            print(
                f"  {signal_name:<8} "
                f"shape={values.shape} "
                f"dtype={values.dtype}"
            )

    labels = data.get("label")

    print("\nLabels:")
    print(f"  shape: {labels.shape}")

    unique_labels = sorted(set(labels))

    print(f"  unique: {unique_labels}")

    print("  counts:")

    for label in unique_labels:
        count = (labels == label).sum()
        print(f"    {label}: {count}")

    print()


def main():
    if not WESAD_ROOT.exists():
        raise FileNotFoundError(
            f"WESAD directory does not exist:\n{WESAD_ROOT}"
        )

    subject_dirs = sorted(
        path
        for path in WESAD_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("S")
    )

    print("=" * 70)
    print("WESAD DATASET INSPECTION")
    print("=" * 70)

    print(f"Dataset path: {WESAD_ROOT}")
    print(f"Subjects found: {len(subject_dirs)}")
    print()

    for subject_dir in subject_dirs:
        inspect_subject(subject_dir)


if __name__ == "__main__":
    main()