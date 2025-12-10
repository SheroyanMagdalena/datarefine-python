import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def apply_standard_scaler(
    df: pd.DataFrame,
    target: str | None = None,
    columns=None,
):
    """
    Applies Standard Scaling (mean=0, std=1) to numeric feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target : str or None
        Name of the target column (will be excluded from scaling if numeric).
    columns : list or None
        Columns to scale. If None, auto-detects numeric columns.

    Returns
    -------
    dict
        {
            "df": scaled DataFrame,
            "scaler": fitted StandardScaler,
            "scaled_columns": list[str]
        }
    """

    new_df = df.copy()

    # Auto-detect numeric columns if none specified
    if columns is None:
        columns = new_df.select_dtypes(include=[np.number]).columns.tolist()

    # Do NOT scale the target column if it is in the list
    if target is not None and target in columns:
        columns.remove(target)

    # If no numeric columns to scale → return unchanged df
    if not columns:
        return {
            "df": new_df,
            "scaler": None,
            "scaled_columns": [],
        }

    # Handle missing values — StandardScaler cannot handle NaN
    new_df[columns] = new_df[columns].fillna(new_df[columns].median())

    # Replace infinite values
    new_df[columns] = new_df[columns].replace([np.inf, -np.inf], np.nan)
    new_df[columns] = new_df[columns].fillna(new_df[columns].median())

    # Fit StandardScaler and transform
    scaler = StandardScaler()
    new_df[columns] = scaler.fit_transform(new_df[columns])

    return {
        "df": new_df,
        "scaler": scaler,
        "scaled_columns": columns,
    }
