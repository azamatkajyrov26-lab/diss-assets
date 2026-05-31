#!/usr/bin/env python3
"""
FastAPI prediction service.
Endpoints:
  GET  /health                    — probe
  POST /api/heatmap               — receive batch of heatmap events
  POST /api/predict/{student_id}  — risk prediction for one student
  POST /api/predict/batch         — risk prediction for all students
  GET  /api/model/info            — model metadata
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import joblib
from pathlib import Path
from datetime import datetime

app = FastAPI(
    title="LMS Prediction Service",
    description="ML-сервис для прогноза академических рисков студентов",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model loading ---
MODEL_PATH = Path(__file__).parent.parent / "data" / "lightgbm_v1.joblib"
model = None
feature_names: List[str] = []


def load_model():
    """Lazy load. In production model is pre-trained and packed into the image."""
    global model, feature_names
    if MODEL_PATH.exists():
        blob = joblib.load(MODEL_PATH)
        model = blob["model"]
        feature_names = blob["feature_names"]


load_model()


# --- Schemas ---
class HeatmapEvent(BaseModel):
    student_id: str
    session_id: str
    page_url: str
    event_type: str  # mousemove | click | scroll | dwell | focus | blur
    viewport_width: int
    viewport_height: int
    timestamp_ms: int
    x: Optional[int] = None
    y: Optional[int] = None
    scroll_depth: Optional[float] = None
    dwell_ms: Optional[int] = None


class HeatmapBatch(BaseModel):
    batch: List[HeatmapEvent]


class PredictResponse(BaseModel):
    student_id: int
    probability: float
    risk_level: str  # low / medium / high
    top_factors: List[dict]


# --- Endpoints ---
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/api/heatmap")
def receive_heatmap(payload: HeatmapBatch):
    # В продакшене — прямая запись в PostgreSQL/Kafka.
    # Для демо просто логируем количество событий.
    print(f"[heatmap] received batch with {len(payload.batch)} events")
    return {"received": len(payload.batch), "status": "accepted"}


@app.post("/api/predict/{student_id}", response_model=PredictResponse)
def predict(student_id: int):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    # Фичи в проде читаются из Redis/PG
    features = _load_features_for(student_id)
    if features is None:
        raise HTTPException(404, f"No data for student {student_id}")
    proba = float(model.predict_proba([features])[0][1])
    level = "high" if proba > 0.7 else ("medium" if proba > 0.42 else "low")
    factors = _top_factors(features, proba)
    return PredictResponse(
        student_id=student_id,
        probability=round(proba, 3),
        risk_level=level,
        top_factors=factors,
    )


@app.get("/api/model/info")
def model_info():
    return {
        "model_type": "LightGBM",
        "version": "1.0.0",
        "trained_on": "145 students, 2023–2024 academic year",
        "metrics": {"F1": 0.824, "ROC_AUC": 0.904, "Accuracy": 0.864},
        "n_features": len(feature_names),
    }


# --- Helpers (in demo — stubs) ---
def _load_features_for(student_id: int):
    feats_path = Path(__file__).parent.parent / "data" / "features.json"
    if not feats_path.exists():
        return None
    with open(feats_path) as f:
        all_feats = json.load(f)
    return all_feats.get(str(student_id))


def _top_factors(features, proba):
    # Заглушка. В проде тут — SHAP.
    return [
        {"feature": "baseline_gpa", "contribution": -0.15},
        {"feature": "attendance_ratio", "contribution": -0.10},
        {"feature": "hesitation_score", "contribution": +0.08},
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
