from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import fetch_california_housing
import numpy as np
import joblib
import os

model = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))

app = FastAPI()

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def root():
    return {
        "message" : "Housing Prcie Predictor API"
    }    
@app.post("/predict")
def predict(features : HouseFeatures):
    data = np.array([[
        np.log(features.MedInc),
        features.HouseAge,
        np.log(features.AveRooms),
        features.AveBedrms,
        np.log(features.Population),
        np.log(features.AveOccup),
        features.Latitude,
        features.Longitude
    ]])

    log_prediction = model.predict(data)
    prediction = np.exp(log_prediction[0])*100000
    return {"predicted_house_value_usd": round(prediction, 2)}

@app.get("/dataset")
def dataset():
    df = fetch_california_housing(as_frame=True).frame
    return df.head(10).to_dict(orient="records")