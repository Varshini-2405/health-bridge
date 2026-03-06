from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="HealthBridge AI Service")

# Load models
RISK_MODEL_PATH = "models/risk_model.pkl"
DISEASE_MODEL_PATH = "models/disease_model.pkl"

if not os.path.exists(RISK_MODEL_PATH) or not os.path.exists(DISEASE_MODEL_PATH):
    raise RuntimeError("Models not found. Run train_model.py first.")

risk_model = joblib.load(RISK_MODEL_PATH)
disease_model = joblib.load(DISEASE_MODEL_PATH)

class SymptomInput(BaseModel):
    fever: int
    cough: int
    fatigue: int
    shortness_of_breath: int
    headache: int
    body_ache: int
    sore_throat: int

@app.get("/")
def read_root():
    return {"message": "HealthBridge AI Intelligence Layer is Online"}

@app.post("/predict")
def predict(input_data: SymptomInput):
    # Convert input to array
    features = np.array([[
        input_data.fever,
        input_data.cough,
        input_data.fatigue,
        input_data.shortness_of_breath,
        input_data.headache,
        input_data.body_ache,
        input_data.sore_throat
    ]])
    
    # Predict Risk
    risk_idx = risk_model.predict(features)[0]
    risk_levels = {0: "Low", 1: "Medium", 2: "High"}
    risk_result = risk_levels.get(int(risk_idx), "Unknown")
    
    # Predict Disease
    disease_result = disease_model.predict(features)[0]
    
    # Generate Recommendations based on risk
    recommendations = []
    if risk_result == "High":
        recommendations = ["Seek immediate medical attention", "Oxygen levels may be low", "Contact nearest hospital"]
    elif risk_result == "Medium":
        recommendations = ["Consult a doctor via tele-consultation", "Rest and stay hydrated", "Monitor symptoms closely"]
    else:
        recommendations = ["Home rest", "Warm fluids", "Continue monitoring temperature"]

    return {
        "risk_level": risk_result,
        "probable_disease": disease_result,
        "recommendations": recommendations,
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
