import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

def train_logistic_regression(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Trains a Logistic Regression classifier on the given dataset.

    - Automatically drops non-numeric feature columns (but reports which)
    - Fills missing values with medians
    - Returns model, scaler, feature_names, metrics, and dropped_non_numeric
    """

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Identify numeric vs non-numeric
    numeric_cols = [
        col for col in X.columns
        if np.issubdtype(X[col].dtype, np.number)
    ]
    non_numeric_cols = [col for col in X.columns if col not in numeric_cols]

    if not numeric_cols:
        raise ValueError(
            "No numeric features available for Logistic Regression. "
            "Please apply encoding / numeric conversion before this step."
        )

    # Keep only numeric features
    X_numeric = X[numeric_cols].copy()

    # Fill missing values with column medians
    X_numeric = X_numeric.fillna(X_numeric.median())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=test_size, random_state=random_state
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "f1": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
    }

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": numeric_cols,
        "dropped_non_numeric": non_numeric_cols,
        "metrics": metrics,
    }
