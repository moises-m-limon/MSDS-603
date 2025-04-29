from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import logging
import uvicorn
import os
import pickle
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="A FastAPI application that predicts diabetes progression based on diagnostic measurements",
    version="1.0.0"
)

# Define the request and response models
class PredictionRequest(BaseModel):
    # Based on the standard diabetes dataset features
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: float

class PredictionResponse(BaseModel):
    # For regression, we return the predicted value directly
    predicted_progression: float

# Simple model class for making predictions
class DiabetesModel:
    def __init__(self):
        # This is a simple linear regression model based on the diabetes dataset
        # These coefficients are approximated for demonstration
        self.coefficients = np.array([
            0.02,    # pregnancies
            0.05,    # glucose (stronger effect)
            0.01,    # blood pressure
            0.01,    # skin thickness
            0.001,   # insulin
            0.05,    # BMI (stronger effect)
            0.3,     # diabetes pedigree function (strongest effect)
            0.03     # age
        ])
        self.intercept = 150.0
    
    def predict(self, X):
        # Simple linear prediction: intercept + weighted sum of features
        return np.array([self.intercept + np.dot(x, self.coefficients) for x in X])

# Load the model
@app.on_event("startup")
async def startup_event():
    global model
    try:
        logger.info("Creating diabetes prediction model")
        # Create the simple model
        model = DiabetesModel()
        logger.info("Model created successfully")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        model = None
        logger.warning("API starting without model. Endpoints will fail until model is available.")

# Health check endpoint
@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to feature array
        features = np.array([[
            request.pregnancies,
            request.glucose,
            request.blood_pressure,
            request.skin_thickness,
            request.insulin,
            request.bmi,
            request.diabetes_pedigree_function,
            request.age
        ]])
        
        # Log the received features
        logger.info(f"Received features: {features}")
        
        # Make prediction
        prediction_result = model.predict(features)
        
        # For regression, we just return the predicted value directly
        predicted_value = float(prediction_result[0])
        
        # Create response
        response = PredictionResponse(
            predicted_progression=predicted_value
        )
        
        logger.info(f"Prediction: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict-batch")
def predict_batch(requests: List[PredictionRequest]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert all requests to a feature matrix
        features = np.array([
            [
                req.pregnancies,
                req.glucose,
                req.blood_pressure,
                req.skin_thickness,
                req.insulin,
                req.bmi,
                req.diabetes_pedigree_function,
                req.age
            ]
            for req in requests
        ])
        
        # Make predictions
        predictions = model.predict(features)
        
        # Create response objects
        results = [
            {
                "predicted_progression": float(pred)
            }
            for pred in predictions
        ]
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)