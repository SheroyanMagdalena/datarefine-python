import pandas as pd
import numpy as np

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values by:
      - Filling numeric columns with median
      - Filling categorical/object columns with mode
      - Safely handling columns with no mode or all NaN values

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame with missing values handled column-wise.
    """

    new_df = df.copy()

    # ---- 1. Handle numeric columns with median ----
    numeric_cols = new_df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        medians = new_df[numeric_cols].median()
        new_df[numeric_cols] = new_df[numeric_cols].fillna(medians)

    # ---- 2. Handle non-numeric columns with mode ----
    non_numeric_cols = new_df.select_dtypes(exclude=[np.number]).columns

    for col in non_numeric_cols:
        col_mode = new_df[col].mode(dropna=True)

        if not col_mode.empty:
            # Use the first mode value
            fill_value = col_mode.iloc[0]
        else:
            # If a column has no mode (e.g., all NaN), fill with empty string or placeholder
            fill_value = ""

        new_df[col] = new_df[col].fillna(fill_value)

    return new_df
