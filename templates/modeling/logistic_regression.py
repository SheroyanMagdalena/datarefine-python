import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_logistic_regression(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Trains a Logistic Regression classifier on the given dataset.

    Returns a JSON-serializable dict:

    {
        "feature_names": list[str],
        "metrics": {
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1": float
        },
        "coefficients": [
            {"feature": str, "coefficient": float}
        ]
    }
    """

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    # Separate target and features
    X = df.drop(columns=[target])
    y = df[target]

    # Ensure all features are numeric
    if not all(np.issubdtype(dt, np.number) for dt in X.dtypes):
        raise ValueError(
            "All features must be numeric before modeling. Apply encoding first."
        )

    # Fill missing values with column medians
    X = X.fillna(X.median())

    feature_names = list(X.columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
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

    # Compute metrics (convert to plain floats)
    metrics_raw = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    metrics = {k: float(v) for k, v in metrics_raw.items()}

    # Coefficients as "feature importances"
    # LogisticRegression.coef_ shape: (n_classes, n_features).
    # For binary classification, we just take the first row.
    if model.coef_.ndim == 2 and model.coef_.shape[0] == 1:
        coef_vector = model.coef_[0]
    else:
        # Multi-class: you can choose another reduction if you want
        # e.g., mean absolute value across classes
        coef_vector = np.mean(np.abs(model.coef_), axis=0)

    coefficients = [
        {"feature": str(name), "coefficient": float(coef)}
        for name, coef in zip(feature_names, coef_vector)
    ]

    return {
        "feature_names": feature_names,
        "metrics": metrics,
        "coefficients": coefficients,
    }
