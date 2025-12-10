import pandas as pd
import numpy as np

def get_numeric_and_categorical_columns(df: pd.DataFrame):
    """
    Splits DataFrame columns into numeric vs categorical groups.
    Datetime columns are returned separately for clarity.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        {
            "numeric": list of numeric column names,
            "categorical": list of categorical/object columns,
            "datetime": list of datetime columns,
            "counts": {
                "numeric": int,
                "categorical": int,
                "datetime": int,
                "total": int
            }
        }
    """

    if df.empty:
        return {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "counts": {
                "numeric": 0,
                "categorical": 0,
                "datetime": 0,
                "total": 0
            }
        }

    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number, "datetime64[ns]", "datetime"]).columns.tolist()

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "counts": {
            "numeric": len(numeric_cols),
            "categorical": len(categorical_cols),
            "datetime": len(datetime_cols),
            "total": df.shape[1]
        }
    }
