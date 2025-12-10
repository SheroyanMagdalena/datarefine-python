import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def apply_minmax_scaler(df: pd.DataFrame, target: str | None = None, columns=None):
    """
    Apply Min-Max scaling to numeric feature columns, keeping the target intact.

    Returns:
        {
            "df": scaled_df,
            "scaled_columns": list_of_scaled_columns,
            "scaler_params": {...}
        }
    """

    # Separate target
    if target is not None and target in df.columns:
        y = df[target]
        X = df.drop(columns=[target])
    else:
        y = None
        X = df

    # Choose numeric columns to scale
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    if not num_cols:
        scaled_df = df.copy()
        return {
            "df": scaled_df,
            "scaled_columns": [],
            "scaler_params": {},
        }

    scaler = MinMaxScaler()
    X_scaled = X.copy()
    X_scaled[num_cols] = scaler.fit_transform(X[num_cols])

    # Reattach target
    if y is not None:
        scaled_df = pd.concat([X_scaled, y], axis=1)
    else:
        scaled_df = X_scaled

    scaler_params = {
        "feature_range": scaler.feature_range,
        "data_min_": scaler.data_min_.tolist(),
        "data_max_": scaler.data_max_.tolist(),
    }

    return {
        "df": scaled_df,
        "scaled_columns": num_cols,
        "scaler_params": scaler_params,
    }
