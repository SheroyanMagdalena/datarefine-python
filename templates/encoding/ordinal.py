ORDINAL = """
from sklearn.preprocessing import OrdinalEncoder

ordinal_cols = df.select_dtypes(include=['object','category']).columns
encoder = OrdinalEncoder()
df[ordinal_cols] = encoder.fit_transform(df[ordinal_cols])
"""