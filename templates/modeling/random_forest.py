import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_random_forest(df: pd.DataFrame,
                        target: str,
                        save_path: str = "feature_importance_rf.png"):
    """
    Trains a Random Forest classifier and returns evaluation results and
    feature importance plot.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features and target.
    target : str
        Name of the target column.
    save_path : str, optional
        Path to save the feature importance PNG.

    Returns
    -------
    dict
        {
            "accuracy": float,
            "report": str,
            "feature_importances": pd.DataFrame,
            "feature_importance_plot": str
        }
    """

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Drop non-numeric columns
    X = X.select_dtypes(include=["number"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Feature importances
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=feat_df)
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return {
        "accuracy": acc,
        "report": report,
        "feature_importances": feat_df,
        "feature_importance_plot": save_path
    }
