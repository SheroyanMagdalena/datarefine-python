import pandas as pd
import numpy as np

def convert_objects_to_numeric(df: pd.DataFrame, threshold: float = 0.7):
    """
    Attempts to convert object columns to numeric if the majority of values 
    are numeric or numeric-like.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    threshold : float, optional
        Minimum proportion of convertible values required (default = 0.7).

    Returns
    -------
    dict
        {
            "df": updated DataFrame,
            "converted_columns": list of columns successfully converted
        }
    """

    new_df = df.copy()
    converted = []

    for col in new_df.columns:
        if new_df[col].dtype != "object":
            continue

        series = new_df[col]

        # Attempt conversion with coercion
        numeric_attempt = pd.to_numeric(series, errors="coerce")

        # Calculate what percentage of values can be converted
        success_rate = numeric_attempt.notna().mean()

        # Convert only if the column is *mostly* numeric-like
        if success_rate >= threshold:
            new_df[col] = numeric_attempt
            converted.append(col)

    return {
        "df": new_df,
        "converted_columns": converted
    }
