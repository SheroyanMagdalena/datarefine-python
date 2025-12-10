import pandas as pd
import numpy as np
import warnings



def normalize_date_formats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempts to normalize date-like columns into pandas datetime format.
    Safely detects date-like columns using heuristics and avoids mis-conversion.

    Rules:
    - Object/string columns: try parsing; only convert if >70% of values parse
    - Integer/float columns: check if values look like timestamps, then convert
    - Leaves non-date columns unchanged

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Copy of df with normalized datetime columns.
    """

    new_df = df.copy()

    for col in new_df.columns:
        series = new_df[col]

        # ---- CASE 1: Numeric timestamps ----
        if pd.api.types.is_numeric_dtype(series):
            # Heuristic: UNIX timestamps typically > 10^9
            if series.dropna().astype(str).str.len().between(10, 13).mean() > 0.7:
                try:
                    new_df[col] = pd.to_datetime(series, unit="s", errors="coerce")
                except:
                    pass
            continue

        # ---- CASE 2: Object/string columns ----
        if pd.api.types.is_object_dtype(series):
            # Try parsing, allowing errors, but inspect success rate
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)

            success_rate = parsed.notna().mean()

            # Only convert if parsing works for most values
            if success_rate > 0.7:  # threshold adjustable
                new_df[col] = parsed

        # ---- CASE 3: Already datetime ----
        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        

    return new_df
