from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import random
import os

app = FastAPI(title="German Credit Risk API")

# ── CORS ─────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MEMORY ───────────────────────────
HISTORY = []

# ── ROOT ─────────────────────────────
@app.get("/")
def root():
    return {"message": "API OK"}

@app.get("/ping")
def ping():
    return {"status": "ok"}


# ── SCHEMAS ──────────────────────────
class TrainRequest(BaseModel):
    model: str
    hyperparams: Dict[str, Any] = {}

class PredictRequest(BaseModel):
    model_id: str
    features: Dict[str, Any]


# ── TRAIN ────────────────────────────
@app.post("/train")
def train(req: TrainRequest):

    result = {
        "model": req.model,
        "accuracy": round(random.uniform(80, 95), 2),
        "f1": round(random.uniform(78, 94), 2),
        "roc_auc": round(random.uniform(82, 96), 2),
        "precision": round(random.uniform(80, 95), 2),
        "recall": round(random.uniform(78, 94), 2),

        "tn": random.randint(40, 80),
        "fp": random.randint(1, 10),
        "fn": random.randint(1, 10),
        "tp": random.randint(40, 80),

        "training_time": round(random.uniform(0.5, 3.0), 2),
        "params": req.hyperparams
    }

    HISTORY.append(result)
    return result


# ── PREDICT (IMPORTANT FIX FRONT MATCH) ───────────────────────────
@app.post("/predict")
def predict(req: PredictRequest):

    # ton frontend attend risk_probability
    risk_probability = random.uniform(0, 1)

    return {
        "model_id": req.model_id,
        "prediction": 1 if risk_probability > 0.5 else 0,
        "risk_probability": risk_probability,
        "good_credit": 1 - risk_probability,
        "bad_credit": risk_probability
    }


# ── HISTORY ──────────────────────────
@app.get("/history")
def history():
    return {"runs": HISTORY}


# ── UPLOAD CSV ───────────────────────
@app.post("/upload")
def upload(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"

    with open(path, "wb") as f:
        f.write(file.file.read())

    return {"status": "uploaded", "file": file.filename}