from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib
import os
from typing import List
import uvicorn

app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = None
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load(os.path.join(MODEL_PATH, "model.joblib"))
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.get("/")
async def root():
    return {"message": "Reddit Model Service"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the CSV file
        df = pd.read_csv(file.file)
        
        # Make predictions
        if model is not None:
            predictions = model.predict(df)
            return {"predictions": predictions.tolist()}
        else:
            return {"error": "Model not loaded"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 