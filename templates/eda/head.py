import pandas as pd

def get_head(df: pd.DataFrame, n: int = 5):
    """
    Returns the first n rows of the dataset in a backend-friendly structure.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    n : int, optional
        Number of rows to return (default = 5).

    Returns
    -------
    dict
        {
            "head": pd.DataFrame,
            "num_rows": int,
            "num_columns": int
        }
    """
    if df.empty:
        return {
            "head": pd.DataFrame(),
            "num_rows": 0,
            "num_columns": df.shape[1]
        }

    return {
        "head": df.head(n),
        "num_rows": df.shape[0],
        "num_columns": df.shape[1]
    }
