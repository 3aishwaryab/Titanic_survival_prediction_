"""FastAPI inference service for Titanic survival predictions."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import sys
# Ensure project root is on sys.path so `src` is importable when running the API with uvicorn.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import joblib
import pandas as pd
from typing import Optional
from src.modeling import load_model, predict_single
from contextlib import asynccontextmanager

MODEL_PATH = "models/best_model.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler to load model on startup and provide clean shutdown point."""
    global MODEL
    try:
        MODEL = load_model(MODEL_PATH)
    except Exception:
        MODEL = None
    yield
    # No explicit shutdown actions needed

app = FastAPI(title="Titanic Inference API", lifespan=lifespan)

class Passenger(BaseModel):
    PassengerId: Optional[int] = Field(None, json_schema_extra={"example": 1})
    Name: str = Field(..., json_schema_extra={"example": "Smith, Mr. John"})
    Sex: str = Field(..., json_schema_extra={"example": "male"})
    Age: float = Field(..., ge=0, le=120, json_schema_extra={"example": 30})
    SibSp: int = Field(..., ge=0, json_schema_extra={"example": 0})
    Parch: int = Field(..., ge=0, json_schema_extra={"example": 0})
    Fare: float = Field(..., ge=0, json_schema_extra={"example": 7.25})
    Pclass: int = Field(..., ge=1, le=3, json_schema_extra={"example": 3})

@app.post("/predict")
def predict(passenger: Passenger):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Use pydantic v2 `model_dump()` to avoid deprecation warnings
    df = pd.DataFrame([passenger.model_dump()])
    try:
        res = predict_single(MODEL, df)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
