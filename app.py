from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model_1.joblib")

app = FastAPI(title="Iris Classifier API")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.on_event("startup")
def load_model():
    global model
    print(1)
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Iris model API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: IrisInput):
    try:
        # Convert single prediction input into pandas DataFrame
        df = pd.DataFrame([data.dict()])  # <-- pandas instead of numpy
        prediction = model.predict(df)[0]
        return {"prediction": str(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
