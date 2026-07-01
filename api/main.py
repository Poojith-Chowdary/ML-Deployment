import joblib
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from src.config import MODEL_PATH, MODEL_VERSION


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", str(MODEL_PATH)))
    print(f"Loading model from {model_path}...")
    app.state.model_artifact = joblib.load(model_path)
    print(f"Model v{app.state.model_artifact['version']} loaded. "
          f"Threshold: {app.state.model_artifact['threshold']}")
    yield
    app.state.model_artifact = None
    print("Model unloaded.")


app = FastAPI(
    title="Employee Churn Prediction API",
    description=(
        "Predicts whether an employee is likely to leave the company. "
        "Built with scikit-learn, FastAPI, Docker, and MLflow."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

from api.routes.predict import router
app.include_router(router)
