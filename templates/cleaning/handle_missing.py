HANDLE_MISSING = """
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(df.mode().iloc[0])
"""