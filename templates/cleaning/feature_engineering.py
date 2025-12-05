FEATURE_ENGINEERING = """
df['total_missing'] = df.isnull().sum(axis=1)
df['numeric_sum'] = df.select_dtypes(include=['int64','float64']).sum(axis=1)
"""