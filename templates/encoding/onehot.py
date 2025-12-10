import pandas as pd


def one_hot_encode(df: pd.DataFrame, target: str | None = None):
    """
    One-hot encode categorical features while keeping the target column intact.

    Returns:
        {
            "df": encoded_df,
            "encoded_columns": list_of_encoded_columns
        }
    """

    # Separate target if provided
    if target is not None and target in df.columns:
        y = df[target]
        X = df.drop(columns=[target])
    else:
        y = None
        X = df

        # If target was provided but missing, we still let pipeline continue;
        # modeling will complain later if needed.

    # Select object/category columns to encode
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if not cat_cols:
        # Nothing to encode
        encoded_df = df.copy()
        return {
            "df": encoded_df,
            "encoded_columns": [],
        }

    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    # Reattach target if we had it
    if y is not None:
        encoded_df = pd.concat([X_encoded, y], axis=1)
    else:
        encoded_df = X_encoded

    return {
        "df": encoded_df,
        "encoded_columns": cat_cols,
    }
