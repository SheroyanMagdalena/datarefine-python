import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def train_xgboost(
    df: pd.DataFrame,
    target: str,
    problem_type: str = "auto",   # "classification", "regression", or "auto"
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    save_path: str = "feature_importance_xgb.png"  # NEW: save feature importance plot
):
    """
    Trains an XGBoost model (classifier or regressor), computes metrics,
    and saves a feature importance plot for reporting.

    Returns a dict compatible with your pipeline's artifact system.
    """

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    # Split features/target
    X = df.drop(columns=[target])
    y = df[target]

    # Ensure all features are numeric
    if not all(np.issubdtype(dt, np.number) for dt in X.dtypes):
        raise ValueError(
            "All features must be numeric before training XGBoost. Apply encoding first."
        )

    # Fill missing values
    X = X.fillna(X.median())

    # Auto-detect problem type
    if problem_type == "auto":
        if not np.issubdtype(y.dtype, np.number):
            resolved_type = "classification"
        else:
            unique_vals = y.nunique(dropna=True)
            if unique_vals <= 20 and y.dropna().apply(float.is_integer).all():
                resolved_type = "classification"
            else:
                resolved_type = "regression"
    else:
        resolved_type = problem_type

    # Train/test split
    stratify = y if resolved_type == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Select model type
    if resolved_type == "classification":
        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            eval_metric="logloss",
            tree_method="hist",
        )
    else:
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            tree_method="hist",
        )

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Compute metrics
    if resolved_type == "classification":
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            "mse": mse,
            "rmse": mean_squared_error(y_test, y_pred, squared=False),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

    # Feature importances
    feature_names = list(X.columns)
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    # ---------- NEW: Save feature importance plot ----------
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title("Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return {
        "model": model,
        "problem_type": resolved_type,
        "feature_names": feature_names,
        "feature_importances": importances,
        "feature_importance_plot": save_path,  # NEW
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
    }
