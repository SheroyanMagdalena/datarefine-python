import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_matrix(df: pd.DataFrame,
                            method: str = "pearson",
                            save_path: str = "correlation_heatmap.png"):
    """
    Computes and plots a correlation matrix heatmap for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    method : str, optional
        Correlation method ("pearson", "spearman", "kendall").
    save_path : str, optional
        Location to save the heatmap PNG image.

    Returns
    -------
    dict
        {
            "correlation_matrix": <pd.DataFrame>,
            "heatmap_path": <str>
        }
    """

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return {
            "correlation_matrix": None,
            "heatmap_path": None,
            "warning": "No numeric columns available for correlation analysis."
        }

    # Compute correlation
    corr_matrix = numeric_df.corr(method=method)

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title(f"Correlation Matrix ({method.capitalize()})")
    plt.tight_layout()

    # Save PNG
    plt.savefig(save_path)
    plt.close()

    return {
        "correlation_matrix": corr_matrix,
        "heatmap_path": save_path
    }
