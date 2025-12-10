import pandas as pd

def one_hot_encode(df: pd.DataFrame, columns=None, drop_first=False):
    """
    Performs one-hot encoding on categorical columns in a safe,
    production-ready manner.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : list or None
        Columns to encode. If None, automatically detects object/category columns.
    drop_first : bool
        Whether to drop the first category to avoid multicollinearity.

    Returns
    -------
    dict
        {
            "df": encoded DataFrame,
            "encoded_columns": list of columns that were encoded,
            "new_columns": list of new one-hot columns generated,
            "category_mapping": dict mapping original_col -> categories
        }
    """

    new_df = df.copy()

    # Auto-detect columns if none provided
    if columns is None:
        columns = new_df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not columns:
        return {
            "df": new_df,
            "encoded_columns": [],
            "new_columns": [],
            "category_mapping": {}
        }

    category_mapping = {}

    # Save category info before encoding
    for col in columns:
        category_mapping[col] = new_df[col].astype("category").cat.categories.tolist()

    # Perform encoding
    encoded_df = pd.get_dummies(new_df, columns=columns, drop_first=drop_first)

    # Collect new column names created
    new_columns = [c for c in encoded_df.columns if any(col + "_" in c for col in columns)]

    return {
        "df": encoded_df,
        "encoded_columns": columns,
        "new_columns": new_columns,
        "category_mapping": category_mapping
    }
