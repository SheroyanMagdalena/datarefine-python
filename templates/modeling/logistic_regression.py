import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_logistic_regression(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    """
    Trains a Logistic Regression classifier on the given dataset.

    Steps:
    - Validate target column
    - Separate X and y
    - Ensure all features are numeric
    - Handle missing values
    - Split train/test sets
    - Scale features
    - Train logistic regression
    - Return model + metrics + scaler + feature names

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing features + target.
    target : str
        Name of the target column.
    test_size : float
        Fraction for test split.
    random_state : int

    Returns
    -------
    dict
        {
            "model": trained LogisticRegression,
            "scaler": StandardScaler,
            "feature_names": list,
            "metrics": {
                "accuracy": float,
                "precision": float,
                "recall": float,
                "f1": float
            }
        }
    """

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    # Separate target and features
    X = df.drop(columns=[target])
    y = df[target]

    # Ensure all features are numeric
    if not all([np.issubdtype(dt, np.number) for dt in X.dtypes]):
        raise ValueError("All features must be numeric before modeling. Apply encoding first.")

    # Fill missing values with column medians
    X = X.fillna(X.median())

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

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "metrics": metrics
    }
