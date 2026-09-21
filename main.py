
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained model
model = joblib.load("Task_4_Champion_Model.joblib")

app = FastAPI(
    title="Titanic ML Prediction API",
    description="Real-Time ML Inference REST API",
    version="1.0.0"
)

# Input data format
class PassengerData(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str
    who: str
    adult_male: bool
    embark_town: str
    alone: bool


@app.get("/")
def home():
    return {
        "message": "Titanic ML Prediction API is running!"
    }


@app.post("/predict")
def predict(data: PassengerData):

    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "prediction_label": "Survived" if prediction == 1 else "Did Not Survive",
        "probability": round(float(probability), 4)
    }
