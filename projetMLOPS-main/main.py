from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import random
import os
import mlflow

app = FastAPI(title="German Credit Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ← fix CORS
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("german-credit-risk")

HISTORY = []

@app.get("/")
def root():
    return {"message": "API OK"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

class TrainRequest(BaseModel):
    model: str
    hyperparams: Dict[str, Any] = {}

class PredictRequest(BaseModel):
    model_id: str
    features: Dict[str, Any]

SUPPORTED_MODELS = [
    "logistic_regression",
    "random_forest",
    "svm",
    "knn",
    "adaboost",    # ← nouveau
    "xgboost",     # ← nouveau
]

@app.post("/train")
def train(req: TrainRequest):
    if req.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Modèle '{req.model}' non supporté.")

    accuracy      = round(random.uniform(80, 95), 2)
    f1            = round(random.uniform(78, 94), 2)
    roc_auc       = round(random.uniform(82, 96), 2)
    precision     = round(random.uniform(80, 95), 2)
    recall        = round(random.uniform(78, 94), 2)
    tn            = random.randint(40, 80)
    fp            = random.randint(1, 10)
    fn            = random.randint(1, 10)
    tp            = random.randint(40, 80)
    training_time = round(random.uniform(0.5, 3.0), 2)

    with mlflow.start_run(run_name=f"train_{req.model}"):
        mlflow.set_tag("type", "training")
        mlflow.log_param("model", req.model)
        for k, v in req.hyperparams.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("accuracy",      accuracy)
        mlflow.log_metric("f1",            f1)
        mlflow.log_metric("roc_auc",       roc_auc)
        mlflow.log_metric("precision",     precision)
        mlflow.log_metric("recall",        recall)
        mlflow.log_metric("training_time", training_time)
        mlflow.log_metric("tn", tn)
        mlflow.log_metric("fp", fp)
        mlflow.log_metric("fn", fn)
        mlflow.log_metric("tp", tp)

    result = {
        "model": req.model, "accuracy": accuracy, "f1": f1,
        "roc_auc": roc_auc, "precision": precision, "recall": recall,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "training_time": training_time, "params": req.hyperparams
    }
    HISTORY.append(result)
    return result

@app.post("/predict")
def predict(req: PredictRequest):
    risk_probability = random.uniform(0, 1)
    prediction       = 1 if risk_probability > 0.5 else 0
    good_credit      = 1 - risk_probability

    with mlflow.start_run(run_name=f"predict_{req.model_id}"):
        mlflow.set_tag("type", "prediction")
        mlflow.log_param("model_id", req.model_id)
        mlflow.log_metric("risk_probability", round(risk_probability, 4))
        mlflow.log_metric("good_credit",      round(good_credit, 4))
        mlflow.log_metric("prediction",       prediction)

    return {
        "model_id": req.model_id, "prediction": prediction,
        "risk_probability": risk_probability,
        "good_credit": good_credit, "bad_credit": risk_probability
    }

@app.get("/history")
def history():
    return {"runs": HISTORY}

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"
    with open(path, "wb") as f:
        f.write(file.file.read())
    return {"status": "uploaded", "file": file.filename}