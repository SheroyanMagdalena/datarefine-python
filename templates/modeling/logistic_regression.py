LOGISTIC_REGRESSION = """
from sklearn.linear_model import LogisticRegression

X = df.drop(target, axis=1)
y = df[target]

model = LogisticRegression(max_iter=500)
model.fit(X, y)

print("Logistic Regression training completed.")
"""