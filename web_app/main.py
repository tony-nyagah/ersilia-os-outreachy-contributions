from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from pathlib import Path

# Define the models directory relative to this file
MODELS_DIR = Path(__file__).parent.parent / "models"

# Initialize FastAPI app
app = FastAPI(
    title="Drug Mutagenicity Predictor",
    description="Predicts mutagenicity of drugs using SMILE strings",
    version="1.0.0",
)


# Pydantic models for request/response
class DrugRequest(BaseModel):
    smiles: str


class PredictionResponse(BaseModel):
    smiles: str
    prediction: str
    probability: float


# Load models at startup
try:
    model_path = MODELS_DIR / "best_model_rf.pkl"
    scaler_path = MODELS_DIR / "feature_scaler.pkl"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Error loading models: {e}")
    raise


@app.get("/")
async def root():
    return {"status": "ok", "message": "Drug Mutagenicity Prediction API"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: DrugRequest):
    try:
        # TODO: Implement feature generation from SMILES
        # This is a placeholder - replace with your actual feature generation
        features = np.random.random((1, len(scaler.get_feature_names_out())))

        # Scale features
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]

        return PredictionResponse(
            smiles=request.smiles,
            prediction="Mutagenic" if prediction == 1 else "Non-mutagenic",
            probability=float(probability),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
