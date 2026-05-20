import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

data = fetch_california_housing(as_frame=True)
df = data.frame

for col in ["MedInc", "AveRooms", "AveOccup", "Population"]:
    df[col] = np.log(df[col])

X = df.drop(columns = ['MedHouseVal'])
y = np.log(df['MedHouseVal'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)    

model = RandomForestRegressor(n_estimators = 100, random_state = 42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"R²   : {r2_score(y_test, y_pred):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

joblib.dump(model, os.path.join(os.path.dirname(__file__), "model.joblib"))
print("Model saved → model.joblib")