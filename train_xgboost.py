import joblib
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def main() -> None:
    """
    Train a basic XGBoost classifier on a sample dataset and save the model.

    Requirements (install first):
        pip install xgboost scikit-learn joblib
    """
    # Load a built-in binary classification dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define a simple XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    # Save the trained model to disk
    model_path = "xgboost_breast_cancer_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved trained model to: {model_path}")


if __name__ == "__main__":
    main()
