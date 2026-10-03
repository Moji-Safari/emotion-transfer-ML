from src.preprocessing.wesad_preprocessing import (
    load_subject,
    create_eda_windows,
)

from src.features.build_features import build_feature_matrix

import numpy as np


def main():
    data = load_subject("S2")

    X, y = create_eda_windows(data)

    feature_matrix = build_feature_matrix(X)

    print("=" * 70)
    print("FEATURE MATRIX TEST")
    print("=" * 70)

    print("Raw windows shape:")
    print(X.shape)

    print("\nFeature matrix shape:")
    print(feature_matrix.shape)

    print("\nLabels shape:")
    print(y.shape)

    print("\nFirst feature vector:")
    print(feature_matrix[0])

    print("\nFirst label:")
    print(y[0])

    print("\nContains NaN:")
    print(np.isnan(feature_matrix).any())

    print("\nContains infinite values:")
    print(np.isinf(feature_matrix).any())


if __name__ == "__main__":
    main()