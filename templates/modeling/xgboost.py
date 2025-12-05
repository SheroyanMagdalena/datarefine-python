XGBOOST = """
from xgboost import XGBClassifier, XGBRegressor

X = df.drop(target, axis=1)
y = df[target]

if y.dtype in ['int64','float64']:
    model = XGBRegressor()
else:
    model = XGBClassifier()

model.fit(X, y)
print("XGBoost model trained.")
"""