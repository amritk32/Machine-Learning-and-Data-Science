import pandas as pd

from fastapi import FastAPI
from predict import predict

app = FastAPI()

@app.get("/")
def home():
    return {"message": "F1 Pit Prediction API Running"}


@app.post("/predict")
def predict_pitstop(data: dict):

    df = pd.DataFrame([data])

    prediction = predict(df)

    return {
        "PitNextLap_Probability": float(prediction[0])
    }