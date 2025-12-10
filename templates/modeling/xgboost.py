import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

from sklearn.model_selection import train_test_split
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

from xgboost import XGBClassifier, XGBRegressor


def train_xgboost(
    df: pd.DataFrame,
    target: str,
    problem_type: str = "auto",   # "classification", "regression", or "auto"
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    save_path: str = "feature_importance_xgb.png",
):
    """
    Trains an XGBoost model (classifier or regressor), computes metrics,
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
        },
        "dropped_non_numeric_features": [str, ...],
        "label_mapping": {original_label: int} | None
    }
    """

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    # Separate features and target
    X = df.drop(columns=[target])
    y_raw = df[target]

    # Drop rows where target is NaN
    mask = y_raw.notna()
    dropped = int((~mask).sum())
    X = X[mask]
    y_raw = y_raw[mask]

    if X.empty or y_raw.empty:
        raise ValueError(
            f"No rows with non-missing target '{target}' remain after cleaning; "
            "cannot train XGBoost."
        )

    # Keep only numeric features, but don't error if we had non-numeric ones.
    numeric_X = X.select_dtypes(include=["number"])
    dropped_non_numeric = [c for c in X.columns if c not in numeric_X.columns]
    X = numeric_X

    if X.shape[1] == 0:
        raise ValueError(
            "No numeric features available for XGBoost after preprocessing. "
            "Make sure encoding/scaling has been applied."
        )

    # Fill remaining numeric NaNs
    X = X.fillna(X.median(numeric_only=True))

    # --- Auto-detect problem type ---
    if problem_type == "auto":
        if not np.issubdtype(y_raw.dtype, np.number):
            resolved_type = "classification"
        else:
            unique_vals = y_raw.nunique(dropna=True)
            if unique_vals <= 20 and y_raw.dropna().apply(float.is_integer).all():
                resolved_type = "classification"
            else:
                resolved_type = "regression"
    else:
        resolved_type = problem_type

    if resolved_type == "classification":
        n_classes = y_raw.nunique(dropna=True)
        if n_classes > 50:
            resolved_type = "regression"

    label_mapping = None  # Will hold {original_label: int} for classification
    

        # --- Auto-detect problem type ---
    if problem_type == "auto":
        if not np.issubdtype(y_raw.dtype, np.number):
            # Non-numeric target → usually classification
            resolved_type = "classification"
        else:
            unique_vals = y_raw.nunique(dropna=True)
            # small integer label set → treat as classification
            if unique_vals <= 20 and y_raw.dropna().apply(float.is_integer).all():
                resolved_type = "classification"
            else:
                resolved_type = "regression"
    else:
        resolved_type = problem_type

    # EXTRA SAFETY:
    # If target is non-numeric but has a small number of categories (e.g. Yes/No),
    # always treat it as classification, even if someone forced regression.
    if not np.issubdtype(y_raw.dtype, np.number):
        n_unique = y_raw.nunique(dropna=True)
        if n_unique <= 50:
            resolved_type = "classification"

    label_mapping = None  # Will hold {original_label: int} for classification

    # --- Prepare y depending on problem type ---
    if resolved_type == "classification":
        # Encode arbitrary labels (strings, etc.) into 0..K-1 for XGBoost
        unique_labels = sorted(pd.unique(y_raw))
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        inv_label_mapping = {v: k for k, v in label_mapping.items()}

        y = y_raw.map(label_mapping)

        # Safety check – all labels must be encodable
        if y.isna().any():
            raise ValueError(
                "Label encoding produced NaNs; check target values in target column "
                f"'{target}'. Example values: {list(y_raw.unique())[:10]!r}"
            )
    else:
        # Regression → ensure numeric target
        if not np.issubdtype(y_raw.dtype, np.number):
            try:
                y = pd.to_numeric(y_raw, errors="raise")
            except Exception:
                raise ValueError(
                    "XGBoost regression requires a numeric target. "
                    f"Found non-numeric values in '{target}', e.g. "
                    f"{list(y_raw.unique())[:10]!r}. "
                    "Use classification instead or encode the target manually."
                )
        else:
            y = y_raw

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
        model = XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
        )
    else:
        model = XGBRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
        )

    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    if resolved_type == "classification":
        # Decode predictions and test labels back to original for metrics/report
        y_test_decoded = pd.Series(y_test).map(inv_label_mapping)
        y_pred_decoded = pd.Series(y_pred).map(inv_label_mapping)

        metrics_raw = {
            "accuracy": accuracy_score(y_test_decoded, y_pred_decoded),
            "precision": precision_score(
                y_test_decoded, y_pred_decoded, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_test_decoded, y_pred_decoded, average="weighted", zero_division=0
            ),
            "f1": f1_score(
                y_test_decoded, y_pred_decoded, average="weighted", zero_division=0
            ),
        }
        report = classification_report(y_test_decoded, y_pred_decoded)
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
    plt.title(f"Feature Importance (XGBoost - {resolved_type})")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    feature_importances = [
        {"feature": str(row["feature"]), "importance": float(row["importance"])}
        for _, row in feat_df.iterrows()
    ]

    # Convert label_mapping keys to str for JSON safety
    label_mapping_json = (
        {str(k): int(v) for k, v in label_mapping.items()}
        if label_mapping is not None
        else None
    )

    return {
        "problem_type": resolved_type,
        "metrics": metrics,
        "report": report,
        "feature_importances": feature_importances,
        "feature_importance_plot": save_path,
        "dropped_rows_with_missing_target": dropped,
        "split_info": split_info,
        "dropped_non_numeric_features": dropped_non_numeric,
        "label_mapping": label_mapping_json,
    }
