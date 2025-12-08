import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/model.pkl")

app = FastAPI(title="ML Model Service")

model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    import numpy as np
    X = np.array([req.features])
    y_pred = model.predict(X)
    return {"prediction": y_pred.tolist()}
