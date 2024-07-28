from fastapi import FastAPI, Body
import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.post("/predict")
async def predict(data: dict = Body(...)):
    features = np.array([data['features']])
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
