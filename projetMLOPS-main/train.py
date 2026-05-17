
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import List
import uvicorn
import io
import pickle
from pathlib import Path
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

app = FastAPI(title="German Credit Risk API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainingData(BaseModel):
    model_type: str
    hyperparameters: dict = {}

class PredictionData(BaseModel):
    features: List[float]

@app.get("/")
def root():
    return {"message": "German Credit Risk API is running ✅"}

@app.post("/train")
def train_model(data: TrainingData):
    # Placeholder for training logic
    # Integrate with your ML training pipeline (Train.py, utils.py, etc.)
    return {
        "status": "training_started",
        "model_type": data.model_type,
        "message": "Model training initiated. Check MLflow for progress."
    }

@app.post("/predict")
def predict(data: PredictionData):
    try:
        # Load latest model (adjust path as needed)
        model_path = "saved_models/logistic_regression_model.pkl"
        scaler_path = "saved_models/logistic_regression_scaler.pkl"
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Reshape and scale input
        features = pd.DataFrame([data.features])
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return {
            "prediction": int(prediction),
            "probability": {"good_credit": float(probability[0]), "bad_credit": float(probability[1])}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
def upload_dataset(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        # Save uploaded file
        upload_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        with open(upload_path, "wb") as f:
            f.write(contents)
        return {"filename": file.filename, "path": upload_path, "status": "uploaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{model_id}")
def download_model(model_id: str):
    model_path = f"saved_models/{model_id}.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {"download_url": f"/models/{model_id}", "model_id": model_id}
MODELS_CONFIG = {
    "logistic_regression": {
        "model":  LogisticRegression(max_iter=200, random_state=42),
        "params": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
    },
    "random_forest": {
        "model":  RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
    },
    "svm": {
        "model":  SVC(probability=True, random_state=42),
        "params": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    },
    "knn": {
        "model":  KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
    },
    "adaboost": {                                           # ← nouveau
        "model":  AdaBoostClassifier(random_state=42),
        "params": {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0, 1.5]},
    },
    "xgboost": {                                            # ← nouveau
        "model":  XGBClassifier(eval_metric="logloss", random_state=42),
        "params": {"n_estimators": [50, 100], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.2]},
    },
}
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
