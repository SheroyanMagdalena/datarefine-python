import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def apply_minmax_scaler(df: pd.DataFrame, columns=None):
    """
    Applies MinMax scaling (0 to 1) to numeric columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : list or None
        Columns to scale. If None, auto-detects numeric columns.

    Returns
    -------
    dict
        {
            "df": scaled DataFrame,
            "scaler": fitted MinMaxScaler,
            "scaled_columns": list[str]
        }
    """

    new_df = df.copy()

    # Auto-detect numeric columns
    if columns is None:
        columns = new_df.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        return {
            "df": new_df,
            "scaler": None,
            "scaled_columns": []
        }

    # Handle missing values — scale requires no NaNs
    new_df[columns] = new_df[columns].fillna(new_df[columns].median())

    # Replace infinite values to avoid scaling crashes
    new_df[columns] = new_df[columns].replace([np.inf, -np.inf], np.nan)
    new_df[columns] = new_df[columns].fillna(new_df[columns].median())

    # Fit scaler
    scaler = MinMaxScaler()
    new_df[columns] = scaler.fit_transform(new_df[columns])

    return {
        "df": new_df,
        "scaler": scaler,
        "scaled_columns": columns
    }
