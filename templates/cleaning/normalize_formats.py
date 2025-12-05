NORMALIZE_FORMATS = """
for col in df.columns:
    try:
        df[col] = pd.to_datetime(df[col])
    except:
        pass
"""