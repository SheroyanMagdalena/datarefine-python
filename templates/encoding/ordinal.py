import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def ordinal_encode(df: pd.DataFrame, columns=None, handle_unknown="use_encoded_value", unknown_value=-1):
    """
    Performs ordinal encoding on selected categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : list or None
        Columns to encode. If None, auto-detects object/category columns.
    handle_unknown : str
        Behavior for unseen categories at transform-time:
          - "use_encoded_value"
          - "error"
    unknown_value : int or float
        Encoding to assign to unseen categories (only used if handle_unknown="use_encoded_value").

    Returns
    -------
    dict
        {
            "df": transformed DataFrame,
            "encoded_columns": list,
            "category_mapping": {column: ordered_categories},
            "encoder": fitted OrdinalEncoder
        }
    """

    new_df = df.copy()

    # Auto-detect columns
    if columns is None:
        columns = new_df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not columns:
        return {
            "df": new_df,
            "encoded_columns": [],
            "category_mapping": {},
            "encoder": None
        }

    # Extract the subset to encode
    subset = new_df[columns]

    # Create encoder
    encoder = OrdinalEncoder(
        handle_unknown=handle_unknown,
        unknown_value=unknown_value
    )

    # Fit + transform
    transformed = encoder.fit_transform(subset)

    # Assign back
    new_df[columns] = transformed

    # Build mapping for interpretability
    category_mapping = {
        col: list(categories)
        for col, categories in zip(columns, encoder.categories_)
    }

    return {
        "df": new_df,
        "encoded_columns": columns,
        "category_mapping": category_mapping,
        "encoder": encoder
    }
