
from pathlib import Path

from fastapi import FastAPI
import joblib


app = FastAPI(title="MLOps Demo API")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {"message": "MLOps ML API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(features: list[float]):
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}
