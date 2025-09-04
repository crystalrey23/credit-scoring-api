from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow import keras

# Muat scaler
scaler = joblib.load("models/credit_scaler.joblib")

# Muat model final dari HDF5
model = keras.models.load_model("models/credit_model_final.h5")

THRESHOLD = 0.60
app = FastAPI(title="Credit Scoring API")

class Features(BaseModel):
    features: list[float]  # 24 nilai numerik

class Prediction(BaseModel):
    probability: float
    label: int

@app.post("/predict", response_model=Prediction)
def predict(data: Features):
    x = np.array(data.features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    prob = float(model.predict(x_scaled)[0][0])
    label = int(prob > THRESHOLD)
    return Prediction(probability=prob, label=label)
