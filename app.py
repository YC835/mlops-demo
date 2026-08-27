
from fastapi import FastAPI
import joblib

app = FastAPI(title="MLOps Demo API")

model = joblib.load("/content/mlops-demo/models/model.pkl")


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
