from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, HTTPException

from .env_vars import MODEL_URI
from .models.churn_predictor import ChurnPredictor

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = ChurnPredictor()
    except Exception as exc:
        app.state.predictor = None
        print(f"Model not available at startup ({MODEL_URI}): {exc}")
    yield
    app.state.predictor = None
    print("Shutting down the API...")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health() -> dict[str, Any]:
    predictor = getattr(app.state, "predictor", None)
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "model_loaded": predictor is not None,
    }


@app.post("/predict")
def predict(payload: Any = Body(...)) -> dict[str, Any]:
    predictor = app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model is not available yet at {MODEL_URI}. "
                "Train/promote a champion model or check the MLflow service."
            ),
        )
    try:
        return predictor.predict(payload)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


