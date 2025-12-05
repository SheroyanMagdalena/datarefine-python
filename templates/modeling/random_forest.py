RANDOM_FOREST = """
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

if y.dtype in ['int64','float64']:
    model = RandomForestRegressor()
else:
    model = RandomForestClassifier()

model.fit(X_train, y_train)
preds = model.predict(X_test)

print("Model performance:")
if y.dtype in ['int64','float64']:
    print("RMSE:", mean_squared_error(y_test, preds, squared=False))
else:
    print("Accuracy:", accuracy_score(y_test, preds))
"""