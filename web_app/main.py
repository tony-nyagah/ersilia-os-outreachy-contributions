from fastapi import FastAPI, HTTPException, Request, Form
from pydantic import BaseModel
import joblib
import numpy as np
import os
from pathlib import Path
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Define the models directory relative to this file
MODELS_DIR = Path(__file__).parent.parent / "models"

# Initialize FastAPI app
app = FastAPI(
    title="Drug Mutagenicity Predictor",
    description="Predicts mutagenicity of drugs using SMILE strings",
    version="1.0.0",
)

# Setup templates
templates = Jinja2Templates(directory="templates")


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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, smiles: str = Form(...)):
    try:
        # TODO: Replace with actual feature generation
        features = np.random.random((1, len(scaler.get_feature_names_out())))

        # Scale features
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "smiles": smiles,
                "prediction": "Mutagenic" if prediction == 1 else "Non-mutagenic",
                "probability": probability,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": f"Prediction failed: {str(e)}"}
        )
