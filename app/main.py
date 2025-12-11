# app/main.py
from typing import List, Optional
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
import joblib
import numpy as np
import os
import time

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
# Optional OTLP exporter (uncomment to enable OTLP export)
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

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

# ---------- OpenTelemetry setup ----------
# Resource identification (service.name useful in traces)
resource = Resource.create({"service.name": "amittech-fraud-detector"})

# set a tracer provider
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# Console exporter (for dev): prints spans to stdout/logs.
console_exporter = ConsoleSpanExporter()
provider.add_span_processor(BatchSpanProcessor(console_exporter))

# Optional: OTLP exporter to send to an OTLP collector (configure endpoint with OTEL_EXPORTER_OTLP_ENDPOINT env var)
# otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
# if otlp_endpoint:
#     otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)  # tweak as needed
#     provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
# ---------- OpenTelemetry setup end ----------

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

# Instrument FastAPI (automatic incoming request spans) - must be after app creation
FastAPIInstrumentor.instrument_app(app)

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
def predict(payload: dict, request: Request):
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

    # predict inside a custom span so we measure model inference time
    probs = None
    try:
        with tracer.start_as_current_span("model.predict") as span:
            t0 = time.time()
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1]
                preds = (probs >= 0.5).astype(int)
            else:
                # fallback to decision_function / predict
                preds = model.predict(X)
                probs = preds.astype(float)
            t1 = time.time()
            
            # Add telemetry attributes
            span.set_attribute("inference.time_ms", (t1 - t0) * 1000.0)
            span.set_attribute("inference.batch_size", int(X.shape[0]))
            span.set_attribute("model.path", str(MODEL_PATH))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # format
    results = []
    for p, prob in zip(preds.tolist(), probs.tolist()):
        results.append({"prediction": int(p), "probability": float(prob)})

    if results and len(results) == 1:
        return results[0]
    return {"predictions": results}
