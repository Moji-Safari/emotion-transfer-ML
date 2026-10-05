from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.preprocessing.wesad_preprocessing import (
    load_subject,
    create_eda_windows,
)

from src.features.build_features import build_feature_matrix
from src.models.random_forest_model import create_random_forest


def main():
    print("=" * 70)
    print("RANDOM FOREST TRAINING TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load WESAD subject
    # ---------------------------------------------------------

    data = load_subject("S2")

    # ---------------------------------------------------------
    # 2. Create EDA windows
    # ---------------------------------------------------------

    X_windows, y = create_eda_windows(data)

    print("\nRaw windows:")
    print(X_windows.shape)

    print("Labels:")
    print(y.shape)

    # ---------------------------------------------------------
    # 3. Extract features
    # ---------------------------------------------------------

    X = build_feature_matrix(X_windows)

    print("\nFeature matrix:")
    print(X.shape)

    # ---------------------------------------------------------
    # 4. Split into training and testing data
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTraining samples:")
    print(len(X_train))

    print("Testing samples:")
    print(len(X_test))

    # ---------------------------------------------------------
    # 5. Create model
    # ---------------------------------------------------------

    model = create_random_forest()

    # ---------------------------------------------------------
    # 6. Train model
    # ---------------------------------------------------------

    print("\nTraining Random Forest...")

    model.fit(X_train, y_train)

    print("Training complete.")

    # ---------------------------------------------------------
    # 7. Make predictions
    # ---------------------------------------------------------

    y_pred = model.predict(X_test)

    # ---------------------------------------------------------
    # 8. Evaluate
    # ---------------------------------------------------------

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

    matrix = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)


if __name__ == "__main__":
    main()