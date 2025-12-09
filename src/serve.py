import os
import traceback

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/model.pkl")

app = FastAPI(title="ML Model Service")

# Load model at startup with logging
try:
    model = joblib.load(MODEL_PATH)
    print(f"[serve] Loaded model from {MODEL_PATH}")
    n_features = getattr(model, "n_features_in_", None)
    print(f"[serve] n_features_in_ = {n_features}")
except Exception:
    print("[serve] Failed to load model:")
    traceback.print_exc()
    raise

class PredictRequest(BaseModel):
    # Adjust list length according to your dataset
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        X = np.array([req.features])

        # Optional but very helpful: check feature length
        expected = getattr(model, "n_features_in_", None)
        if expected is not None and X.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} features, got {X.shape[1]} "
                f"(features={req.features})"
            )

        y_pred = model.predict(X)
        return {"prediction": y_pred.tolist()}
    except Exception as e:
        print("[serve] Error during prediction:")
        traceback.print_exc()
        # Return a clear error instead of generic 500
        raise HTTPException(status_code=400, detail=str(e))
