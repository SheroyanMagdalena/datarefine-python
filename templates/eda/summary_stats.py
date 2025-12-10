import pandas as pd
import numpy as np

def get_summary_stats(df: pd.DataFrame):
    """
    Computes comprehensive summary statistics for numeric, categorical,
    and datetime columns in a backend-friendly format.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        {
            "numeric_summary": DataFrame,
            "categorical_summary": DataFrame,
            "datetime_summary": DataFrame,
            "counts": {
                "numeric_columns": int,
                "categorical_columns": int,
                "datetime_columns": int
            }
        }
    """

    if df.empty:
        return {
            "numeric_summary": pd.DataFrame(),
            "categorical_summary": pd.DataFrame(),
            "datetime_summary": pd.DataFrame(),
            "counts": {
                "numeric_columns": 0,
                "categorical_columns": 0,
                "datetime_columns": 0
            }
        }

    # Identify dtypes
    numeric_cols = df.select_dtypes(include=[np.number])
    categorical_cols = df.select_dtypes(include=["object", "category"])
    datetime_cols = df.select_dtypes(include=["datetime", "datetime64[ns]"])

    # Compute summaries
    numeric_summary = numeric_cols.describe().T if not numeric_cols.empty else pd.DataFrame()
    categorical_summary = categorical_cols.describe().T if not categorical_cols.empty else pd.DataFrame()

    # For datetime columns: min/max, number of missing, etc.
    if not datetime_cols.empty:
        datetime_summary = pd.DataFrame({
            "min": datetime_cols.min(),
            "max": datetime_cols.max(),
            "missing": datetime_cols.isna().sum()
        })
    else:
        datetime_summary = pd.DataFrame()

    return {
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "datetime_summary": datetime_summary,
        "counts": {
            "numeric_columns": numeric_cols.shape[1],
            "categorical_columns": categorical_cols.shape[1],
            "datetime_columns": datetime_cols.shape[1]
        }
    }
