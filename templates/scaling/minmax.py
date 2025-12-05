MINMAX_SCALER = """
from sklearn.preprocessing import MinMaxScaler

num_cols = df.select_dtypes(include=['int64','float64']).columns
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
"""