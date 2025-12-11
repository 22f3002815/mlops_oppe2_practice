# app/main.py
from typing import List, Optional
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import os

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/model.pkl")

class Transaction(BaseModel):
    Time: Optional[float] = None
    Amount: float
    # Accept anonymized V1..V28 — optional
    V1: Optional[float] = None
    V2: Optional[float] = None
    V3: Optional[float] = None
    V4: Optional[float] = None
    V5: Optional[float] = None
    V6: Optional[float] = None
    V7: Optional[float] = None
    V8: Optional[float] = None
    V9: Optional[float] = None
    V10: Optional[float] = None
    V11: Optional[float] = None
    V12: Optional[float] = None
    V13: Optional[float] = None
    V14: Optional[float] = None
    V15: Optional[float] = None
    V16: Optional[float] = None
    V17: Optional[float] = None
    V18: Optional[float] = None
    V19: Optional[float] = None
    V20: Optional[float] = None
    V21: Optional[float] = None
    V22: Optional[float] = None
    V23: Optional[float] = None
    V24: Optional[float] = None
    V25: Optional[float] = None
    V26: Optional[float] = None
    V27: Optional[float] = None
    V28: Optional[float] = None

# Global model variable
model = None
FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    
    yield  # Application runs here
    
    # Shutdown: Cleanup (if needed)
    print("Shutting down application...")
    model = None

# Create FastAPI app with lifespan
app = FastAPI(title="AmitTech Fraud Detector", lifespan=lifespan)

def tx_to_vector(tx: Transaction):
    # create vector for FEATURE_ORDER; missing -> 0.0
    vec = []
    d = tx.dict()
    for f in FEATURE_ORDER:
        vec.append(d.get(f, 0.0))
    return np.array(vec, dtype=float)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Fraud Detector API is running"}

@app.post("/predict")
def predict(payload: dict):
    """
    Accepts:
      - {"transaction": {...}} for single
      - {"transactions": [{...}, {...}, ...]} for batch
    Returns: predictions and probabilities
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # allow both single and batch
    if "transaction" in payload:
        tx = Transaction(**payload["transaction"])
        X = tx_to_vector(tx).reshape(1, -1)
    elif "transactions" in payload:
        txs = [Transaction(**t) for t in payload["transactions"]]
        X = np.vstack([tx_to_vector(t) for t in txs])
    else:
        # try if payload itself is a transaction map
        try:
            tx = Transaction(**payload)
            X = tx_to_vector(tx).reshape(1, -1)
        except Exception:
            raise HTTPException(status_code=400, detail="Payload format invalid")

    # predict
    probs = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= 0.5).astype(int)
        else:
            # fallback to decision_function / predict
            preds = model.predict(X)
            probs = preds.astype(float)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # format
    results = []
    for p, prob in zip(preds.tolist(), probs.tolist()):
        results.append({"prediction": int(p), "probability": float(prob)})

    if results and len(results) == 1:
        return results[0]
    return {"predictions": results}
