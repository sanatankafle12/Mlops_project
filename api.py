from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the FastAPI app
app = FastAPI()

# Define the input data model
class Features(BaseModel):
    features: list = Field(..., example=[5.1, 3.5, 1.4, 0.2])

# Define the prediction endpoint
@app.post("/predict")
async def predict(features: Features):
    data = np.array([features.features])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
