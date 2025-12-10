import numpy as np
import pandas as pd

def detect_outliers(df: pd.DataFrame, threshold: float = 3.0):
    """
    Detect outliers in a DataFrame using Z-score method.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    threshold : float, optional
        Z-score threshold for marking outliers. Default is 3.0.

    Returns
    -------
    dict
        {
          "outliers_per_column": pd.Series,
          "total_outliers": int,
          "outlier_mask": pd.DataFrame (bool)
        }
    """

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return {
            "outliers_per_column": pd.Series(dtype=int),
            "total_outliers": 0,
            "outlier_mask": pd.DataFrame(False, index=df.index, columns=df.columns),
            "message": "No numeric columns available for outlier detection."
        }

    # Compute means and std, replacing std=0 with NaN to avoid division by zero
    means = numeric_df.mean()
    stds = numeric_df.std(ddof=0).replace(0, np.nan)

    # Compute z-scores
    z_scores = (numeric_df - means) / stds

    # Absolute z-score for threshold comparison
    outlier_mask = np.abs(z_scores) > threshold

    # Count outliers per column (sorted)
    outliers_per_column = outlier_mask.sum().sort_values(ascending=False)

    total_outliers = int(outlier_mask.sum().sum())

    return {
        "outliers_per_column": outliers_per_column,
        "total_outliers": total_outliers,
        "outlier_mask": outlier_mask
    }
