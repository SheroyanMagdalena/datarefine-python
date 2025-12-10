import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def ordinal_encode(
    df: pd.DataFrame,
    target: str | None = None,
    columns: list[str] | None = None,
):
    """
    Ordinal-encode categorical features while keeping the target column intact.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features (and possibly target).
    target : str or None
        Name of the target column (will be preserved, not encoded).
    columns : list[str] or None
        Specific columns to ordinal-encode. If None, auto-detects object/category columns.

    Returns
    -------
    dict
        {
            "df": encoded_df,
            "encoder": fitted OrdinalEncoder,
            "encoded_columns": list[str]
        }
    """

    new_df = df.copy()

    # Separate target if provided
    if target is not None and target in new_df.columns:
        y = new_df[target]
        X = new_df.drop(columns=[target])
    else:
        y = None
        X = new_df

    # Determine columns to encode
    if columns is None:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    else:
        cat_cols = columns

    if not cat_cols:
        # Nothing to encode
        return {
            "df": df.copy(),
            "encoder": None,
            "encoded_columns": [],
        }

    # Fit OrdinalEncoder on categorical columns
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cat_cols] = enc.fit_transform(X[cat_cols])

    # Reattach target if we had it
    if y is not None:
        encoded_df = pd.concat([X, y], axis=1)
    else:
        encoded_df = X

    return {
        "df": encoded_df,
        "encoder": enc,
        "encoded_columns": cat_cols,
    }
