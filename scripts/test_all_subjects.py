import numpy as np

from src.data.build_dataset import build_all_subjects_dataset


def main():
    print("=" * 70)
    print("BUILDING COMPLETE WESAD DATASET")
    print("=" * 70)

    X, y, groups = build_all_subjects_dataset()

    print("\n" + "=" * 70)
    print("FINAL DATASET")
    print("=" * 70)

    print("Feature matrix shape:")
    print(X.shape)

    print("\nLabels shape:")
    print(y.shape)

    print("\nGroups shape:")
    print(groups.shape)

    print("\nTotal baseline windows:")
    print(np.sum(y == 0))

    print("\nTotal stress windows:")
    print(np.sum(y == 1))

    print("\nSubjects:")
    print(np.unique(groups))

    print("\nNaN:")
    print(np.isnan(X).any())

    print("\nInfinite:")
    print(np.isinf(X).any())


if __name__ == "__main__":
    main()