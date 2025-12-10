import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
)


def train_random_forest(
    df: pd.DataFrame,
    target: str,
    problem_type: str = "auto",   # "classification", "regression", or "auto"
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    save_path: str = "feature_importance_rf.png",
):
    """
    Trains a Random Forest model (classifier or regressor), computes metrics,
    and saves a feature importance plot.

    Returns JSON-serializable artifacts:

    {
        "problem_type": "classification" | "regression",
        "metrics": { ... },
        "report": str | None,
        "feature_importances": [
            {"feature": str, "importance": float},
            ...
        ],
        "feature_importance_plot": str,
        "dropped_rows_with_missing_target": int,
        "split_info": {
            "stratified": bool,
            "reason": str | None,
            "class_counts": {class_label: count, ...} | None
        }
    }
    """

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Drop rows where target is NaN
    mask = y.notna()
    dropped = int((~mask).sum())
    X = X[mask]
    y = y[mask]

    if X.empty or y.empty:
        raise ValueError(
            f"No rows with non-missing target '{target}' remain after cleaning; "
            "cannot train RandomForest."
        )

    # Keep only numeric features
    X = X.select_dtypes(include=["number"])

    if X.shape[1] == 0:
        raise ValueError(
            "No numeric features available for RandomForest after preprocessing. "
            "Make sure encoding/scaling has been applied."
        )

    # Fill remaining numeric NaNs
    X = X.fillna(X.median(numeric_only=True))

    # --- Auto-detect problem type ---
    if problem_type == "auto":
        if not np.issubdtype(y.dtype, np.number):
            resolved_type = "classification"
        else:
            unique_vals = y.nunique(dropna=True)
            # treat as classification if small integer label set (e.g. 0/1 or 1..5)
            if unique_vals <= 20 and y.dropna().apply(float.is_integer).all():
                resolved_type = "classification"
            else:
                resolved_type = "regression"
    else:
        resolved_type = problem_type

    # --- Decide whether we can safely stratify ---
    stratify = None
    split_info = {
        "stratified": False,
        "reason": None,
        "class_counts": None,
    }

    if resolved_type == "classification":
        class_counts = y.value_counts()
        min_count = int(class_counts.min())

        # Save counts in a JSON-safe way (keys -> strings)
        split_info["class_counts"] = {str(k): int(v) for k, v in class_counts.items()}

        if min_count >= 2:
            # Safe to stratify
            stratify = y
            split_info["stratified"] = True
            split_info["reason"] = "all_classes_have_at_least_2_samples"
        else:
            # Too few samples in at least one class -> cannot stratify
            stratify = None
            split_info["stratified"] = False
            split_info["reason"] = (
                "disabled_stratify_min_class_count_lt_2"
            )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Choose model
    if resolved_type == "classification":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
        )

    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    if resolved_type == "classification":
        metrics_raw = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        report = classification_report(y_test, y_pred)
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)

        metrics_raw = {
            "mse": mse,
            "rmse": rmse,
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }
        report = None

    metrics = {k: float(v) for k, v in metrics_raw.items()}

    # Feature importances
    feat_df = (
        pd.DataFrame(
            {"feature": X.columns, "importance": model.feature_importances_}
        )
        .sort_values(by="importance", ascending=False)
    )

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=feat_df)
    plt.title(f"Feature Importance (Random Forest - {resolved_type})")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    feature_importances = [
        {"feature": str(row["feature"]), "importance": float(row["importance"])}
        for _, row in feat_df.iterrows()
    ]

    return {
        "problem_type": resolved_type,
        "metrics": metrics,
        "report": report,
        "feature_importances": feature_importances,
        "feature_importance_plot": save_path,
        "dropped_rows_with_missing_target": dropped,
        "split_info": split_info,
    }
