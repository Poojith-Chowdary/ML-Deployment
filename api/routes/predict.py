import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.schemas import PredictRequest, PredictResponse, HealthResponse, ModelInfoResponse
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES, MODEL_VERSION

router = APIRouter()


@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(payload: PredictRequest, request: Request):
    artifact = request.app.state.model_artifact
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]

    row = payload.model_dump()
    df = pd.DataFrame([row])[ALL_FEATURES]

    prob = float(pipeline.predict_proba(df)[0][1])
    label = "Churn" if prob >= threshold else "Retain"

    return PredictResponse(
        prediction=label,
        probability=round(prob, 4),
        model_version=MODEL_VERSION,
        threshold_used=threshold,
    )


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health(request: Request):
    return HealthResponse(
        status="ok",
        model_loaded=request.app.state.model_artifact is not None,
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
def model_info(request: Request):
    artifact = request.app.state.model_artifact
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_version=artifact["version"],
        decision_threshold=artifact["threshold"],
        feature_count=len(ALL_FEATURES),
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
