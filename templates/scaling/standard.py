STANDARD_SCALER = """
from sklearn.preprocessing import StandardScaler

num_cols = df.select_dtypes(include=['int64','float64']).columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
"""