from src.preprocessing.wesad_preprocessing import (
    load_subject,
    create_eda_windows,
)

from src.features.eda_features import extract_eda_features


def main():

    data = load_subject("S2")

    X, y = create_eda_windows(data)

    print("=" * 70)
    print("EDA FEATURE EXTRACTION TEST")
    print("=" * 70)

    print("Windows:", X.shape)
    print("Labels:", y.shape)

    first_window = X[0]

    features = extract_eda_features(first_window)

    print("\nFeatures from first window:")
    print("-" * 70)

    for name, value in features.items():
        print(f"{name:<25} {value:.6f}")


if __name__ == "__main__":
    main()