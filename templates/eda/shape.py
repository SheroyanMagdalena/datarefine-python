import pandas as pd

def get_shape(df: pd.DataFrame):
    """
    Returns the shape (rows, columns) of the DataFrame,
    along with useful metadata for UI/EDA dashboards.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        {
            "rows": int,
            "columns": int,
            "shape": tuple,
            "is_empty": bool
        }
    """
    rows, cols = df.shape

    return {
        "rows": rows,
        "columns": cols,
        "shape": (rows, cols),
        "is_empty": rows == 0
    }
