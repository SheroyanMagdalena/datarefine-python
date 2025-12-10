import pandas as pd

def remove_duplicates(df: pd.DataFrame, subset=None, keep="first", reset_index=True):
    """
    Removes duplicate rows from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    subset : list or str, optional
        Columns to consider when identifying duplicates.
        If None, all columns are used (default pandas behavior).
    keep : {"first", "last", False}, optional
        - "first": keep first occurrence, remove others
        - "last": keep last occurrence
        - False: remove all duplicates
    reset_index : bool, optional
        Whether to reset the DataFrame index after removing duplicates.

    Returns
    -------
    dict
        {
           "df": cleaned DataFrame,
           "duplicates_removed": int
        }
    """

    original_len = len(df)

    new_df = df.drop_duplicates(subset=subset, keep=keep)

    if reset_index:
        new_df = new_df.reset_index(drop=True)

    removed = original_len - len(new_df)

    return {
        "df": new_df,
        "duplicates_removed": removed
    }
