"""
serve.py — FastAPI model serving voi blue-green support va strict data validation.

Startup: load model tu MLflow Registry alias 'production'.
Endpoints:
  POST /predict               -- score mot batch features
  GET  /health/active-version -- version hien tai dang serve
  POST /reload                -- reload model tu registry (sau khi swap alias)
  GET  /metrics               -- Prometheus metrics endpoint

Data Validation Rules (enforced per row):
  - latency_p99: float, > 0         (negative/zero latency is impossible)
  - error_rate:  float, 0.0 to 1.0  (fraction, not percentage)
  - rps:         float, >= 0         (non-negative throughput)

Usage:
    export MLFLOW_TRACKING_URI=http://localhost:5000
    uv run python serve.py
    uv run python serve.py --host 0.0.0.0 --port 8000
"""

import argparse
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

import mlflow
import mlflow.sklearn
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, field_validator, model_validator

logging.basicConfig(level=logging.INFO, format="[serve] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────── Prometheus Metrics ───────────────────────────────
_serve_requests = Counter("serve_requests_total", "Total predict requests")
_serve_errors = Counter("serve_errors_total", "Total predict errors (validation + model)")
_serve_latency = Histogram(
    "serve_predict_latency_seconds",
    "Predict endpoint latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
_serve_active_version = Gauge("serve_active_version", "Currently loaded model version number")
_serve_anomaly_rate = Gauge("serve_batch_anomaly_rate", "Fraction of anomalies in last predict batch")

MODEL_NAME = "anomaly-detector"
MODEL_URI = f"models:/{MODEL_NAME}@production"
FEATURES = ["latency_p99", "error_rate", "rps"]

# Global model state
_state: dict[str, Any] = {
    "model": None,
    "version": None,
    "model_uri": None,
}


# ─────────────────────────── Pydantic Schemas ────────────────────────────────

class FeatureRow(BaseModel):
    """Single row of features with strict business-rule validation."""
    latency_p99: Annotated[float, Field(gt=0, description="p99 latency in ms — must be positive")]
    error_rate: Annotated[float, Field(ge=0.0, le=1.0, description="Error rate as a fraction [0, 1]")]
    rps: Annotated[float, Field(ge=0.0, description="Requests per second — must be non-negative")]


class PredictRequest(BaseModel):
    """Prediction request: list of feature rows.

    Accepts either:
      - Structured: {"features": [{"latency_p99": 120, "error_rate": 0.01, "rps": 450}]}
      - Raw matrix: {"features": [[120, 0.01, 450]]}  (order: latency_p99, error_rate, rps)
    """
    features: list[list[float]] | list[FeatureRow]

    @model_validator(mode="before")
    @classmethod
    def normalise_features(cls, data: Any) -> Any:
        """Normalise both raw list and structured dict inputs into structured rows."""
        feats = data.get("features", []) if isinstance(data, dict) else []
        if not feats:
            return data
        # If items are plain lists, convert to dicts for FeatureRow validation
        if isinstance(feats[0], (list, tuple)):
            if any(len(row) != 3 for row in feats):
                raise ValueError(
                    f"Each feature row must have exactly 3 values "
                    f"[latency_p99, error_rate, rps], got lengths: {[len(r) for r in feats]}"
                )
            data["features"] = [
                {"latency_p99": row[0], "error_rate": row[1], "rps": row[2]}
                for row in feats
            ]
        return data

    @field_validator("features")
    @classmethod
    def must_not_be_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("features list must not be empty")
        return v


class PredictResponse(BaseModel):
    predictions: list[int]    # -1 = anomaly, 1 = normal
    scores: list[float]       # raw anomaly score (more negative = more anomalous)
    anomaly_rate: float       # fraction of anomalies in this batch
    version: str
    model_name: str


class VersionResponse(BaseModel):
    model_name: str
    version: str
    alias: str
    model_uri: str


# ─────────────────────────── Model Loading ───────────────────────────────────

def _load_model() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    alias_mv = client.get_model_version_by_alias(MODEL_NAME, "production")

    model = mlflow.sklearn.load_model(MODEL_URI)
    _state["model"] = model
    _state["version"] = alias_mv.version
    _state["model_uri"] = MODEL_URI
    log.info("Loaded %s v%s from alias 'production'", MODEL_NAME, alias_mv.version)
    try:
        _serve_active_version.set(int(alias_mv.version))
    except (ValueError, TypeError):
        pass


# ─────────────────────────── App Lifecycle ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    _state["model"] = None


app = FastAPI(
    title="Anomaly Detector API",
    description=(
        "Serve IsolationForest anomaly detection model with strict input validation. "
        "Features: latency_p99 (ms, >0), error_rate ([0,1]), rps (>=0)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ─────────────────────────── Endpoints ───────────────────────────────────────

@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics for scraping."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Score a batch of feature rows.

    Validates business constraints before inference:
    - latency_p99 > 0
    - 0 <= error_rate <= 1
    - rps >= 0
    """
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    _serve_requests.inc()
    t0 = time.perf_counter()
    try:
        # Convert FeatureRow objects to numpy array
        X = np.array([
            [row.latency_p99, row.error_rate, row.rps]
            for row in req.features
        ])

        predictions = _state["model"].predict(X).tolist()
        scores = _state["model"].score_samples(X).tolist()
        anomaly_rate = sum(1 for p in predictions if p == -1) / len(predictions)

        _serve_anomaly_rate.set(anomaly_rate)
        _serve_latency.observe(time.perf_counter() - t0)

        return PredictResponse(
            predictions=predictions,
            scores=scores,
            anomaly_rate=anomaly_rate,
            version=str(_state["version"]),
            model_name=MODEL_NAME,
        )
    except Exception as exc:
        _serve_errors.inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health/active-version", response_model=VersionResponse)
def active_version():
    """Return currently loaded model version info."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return VersionResponse(
        model_name=MODEL_NAME,
        version=str(_state["version"]),
        alias="production",
        model_uri=str(_state["model_uri"]),
    )


@app.get("/health")
def health():
    """Simple liveness probe."""
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.post("/reload")
def reload():
    """Reload model from registry — call after alias 'production' is swapped."""
    try:
        _load_model()
        return {"status": "reloaded", "version": str(_state["version"])}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────── CLI ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run anomaly detector API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload-on-start", action="store_true", default=False)
    args = parser.parse_args()

    if args.reload_on_start:
        uvicorn.run("serve:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
