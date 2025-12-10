import pandas as pd
import numpy as np

def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered feature columns to the DataFrame:

    - total_missing: number of missing values per row
    - percent_missing: percentage of missing values per row (0–100)
    - numeric_sum: sum of numeric values per row
    - numeric_mean: mean of numeric values per row
    - row_quality_score: simple data-quality score per row (1 - missing_fraction)
    - has_missing: True if the row has at least one missing value
    - all_missing: True if all values in the row are missing
    - any_negative_numeric: True if any numeric value in the row is negative

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        A copy of df with new engineered feature columns.
    """

    new_df = df.copy()

    # Total columns (for percent_missing)
    total_columns = new_df.shape[1] if new_df.shape[1] > 0 else 1  # avoid division by zero

    # 1. Missing value features
    total_missing = new_df.isnull().sum(axis=1)
    percent_missing = (total_missing / total_columns) * 100.0

    new_df["total_missing"] = total_missing
    new_df["percent_missing"] = percent_missing

    # 2. Numeric-based features
    numeric_df = new_df.select_dtypes(include=[np.number])

    if not numeric_df.empty:
        numeric_sum = numeric_df.sum(axis=1)
        numeric_mean = numeric_df.mean(axis=1)

        any_negative_numeric = (numeric_df < 0).any(axis=1)
    else:
        # No numeric columns: fill with safe defaults
        numeric_sum = pd.Series(0, index=new_df.index)
        numeric_mean = pd.Series(0.0, index=new_df.index)
        any_negative_numeric = pd.Series(False, index=new_df.index)

    new_df["numeric_sum"] = numeric_sum
    new_df["numeric_mean"] = numeric_mean

    # 3. Row-wise quality score (1 = no missing, 0 = all missing)
    missing_fraction = total_missing / total_columns
    row_quality_score = 1.0 - missing_fraction
    new_df["row_quality_score"] = row_quality_score.clip(lower=0.0, upper=1.0)

    # 4. Boolean flag columns
    new_df["has_missing"] = total_missing > 0
    new_df["all_missing"] = total_missing == total_columns
    new_df["any_negative_numeric"] = any_negative_numeric

    return new_df
