# Employee Churn Prediction API

> End-to-end ML deployment: scikit-learn model served via FastAPI, containerised with Docker, experiment-tracked with MLflow, tested with pytest, and auto-deployed to Render via GitHub Actions.

[![CI](https://github.com/Poojith-Chowdary/ml-deployment/actions/workflows/ci.yml/badge.svg)](https://github.com/Poojith-Chowdary/ml-deployment/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker)](./Dockerfile)
[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)](./requirements.txt)

**Live endpoint:** `https://ml-churn-api-8zd5.onrender.com`

---

## What this project demonstrates

| Skill | Implementation |
|---|---|
| ML model training | `RandomForestClassifier` with threshold tuning for class imbalance |
| REST API | FastAPI with Pydantic validation, auto-docs, and lifespan model loading |
| Containerisation | Multi-stage Dockerfile, docker-compose with MLflow service |
| Experiment tracking | MLflow logging — params, metrics, and model artifact per run |
| Testing | 36 pytest tests across API, model, and preprocessing (67% coverage) |
| CI/CD | GitHub Actions — lint, test, docker build, and auto-deploy on merge to main |
| Deployment | Render.com free tier via Docker, auto-redeploys on every push to main |

---

## Architecture

```
data/raw/CSV
     │
     ▼
src/train.py  ──────►  models/model.joblib
     │                       │
     ▼                       ▼
MLflow tracking        api/main.py (FastAPI)
(mlflow.db)                  │
                       ┌─────┴──────┐
                       │            │
                  POST /predict   GET /health
                       │         GET /model-info
                       ▼
                  JSON response
                  { prediction, probability,
                    model_version, threshold_used }
```

---

## Quick start (local)

```bash
git clone https://github.com/Poojith-Chowdary/ml-deployment.git
cd ml-deployment
pip install -r requirements.txt

python generate_data.py    # creates data/raw/dataset.csv
python src/train.py        # trains model → models/model.joblib
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### Or run with Docker

```bash
docker compose up --build
```

- API: http://localhost:8000/docs
- MLflow UI: http://localhost:5000

---

## Sample request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 28,
    "BusinessTravel": "Travel_Frequently",
    "DailyRate": 200,
    "Department": "Sales",
    "DistanceFromHome": 25,
    "Education": 2,
    "EducationField": "Marketing",
    "EnvironmentSatisfaction": 1,
    "Gender": "Male",
    "HourlyRate": 35,
    "JobInvolvement": 1,
    "JobLevel": 1,
    "JobRole": "Sales Representative",
    "JobSatisfaction": 1,
    "MaritalStatus": "Single",
    "MonthlyIncome": 1500,
    "MonthlyRate": 3000,
    "NumCompaniesWorked": 7,
    "OverTime": "Yes",
    "PercentSalaryHike": 11,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 1,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 3,
    "TrainingTimesLastYear": 0,
    "WorkLifeBalance": 1,
    "YearsAtCompany": 1,
    "YearsInCurrentRole": 0,
    "YearsSinceLastPromotion": 0,
    "YearsWithCurrManager": 0
  }'
```

**Response:**
```json
{
  "prediction": "Churn",
  "probability": 0.2865,
  "model_version": "1.0.0",
  "threshold_used": 0.3
}
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Returns churn prediction + probability |
| `GET` | `/health` | Liveness check — confirms model is loaded |
| `GET` | `/model-info` | Version, threshold, feature list |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Model card

| Property | Value |
|---|---|
| Algorithm | `RandomForestClassifier` (scikit-learn) |
| Dataset | IBM HR Employee Attrition — 1,470 employees, 30 features |
| Churn rate | ~10% (class imbalance handled via `class_weight="balanced"`) |
| Accuracy | 0.79 |
| F1 (churn class) | 0.40 |
| AUC-ROC | 0.79 |
| Decision threshold | 0.30 (tuned to maximise F1, not default 0.5) |
| Top predictors | OverTime, JobSatisfaction, WorkLifeBalance, MonthlyIncome, NumCompaniesWorked |
| Known limitations | Trained on synthetic data; real-world performance will vary |

---

## Project structure

```
ml-deployment/
├── src/
│   ├── config.py          # paths, feature lists, constants
│   ├── preprocess.py      # sklearn ColumnTransformer pipeline
│   └── train.py           # training script with MLflow logging
├── api/
│   ├── main.py            # FastAPI app + lifespan model loading
│   ├── schemas.py         # Pydantic request/response models
│   └── routes/
│       └── predict.py     # /predict, /health, /model-info
├── tests/
│   ├── conftest.py        # shared fixtures (TestClient, payloads)
│   ├── test_api.py        # 18 API integration tests
│   ├── test_model.py      # 8 model unit tests
│   └── test_preprocess.py # 10 preprocessing unit tests
├── models/
│   └── model.joblib       # trained pipeline artifact
├── .github/workflows/
│   ├── ci.yml             # test + lint + docker build on every push
│   └── deploy.yml         # trigger Render deploy on merge to main
├── Dockerfile             # multi-stage, python:3.11-slim
├── docker-compose.yml     # api + mlflow services
├── render.yaml            # Render Blueprint config
└── generate_data.py       # synthetic IBM HR dataset generator
```

---

## Running tests

```bash
pip install -r requirements-dev.txt
set PYTHONPATH=.        # Windows
# export PYTHONPATH=.  # Mac/Linux
pytest tests/ -v
```

Expected: **36 passed**

---

## Deployment (Render)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Blueprint**
3. Connect your GitHub repo — Render reads `render.yaml` automatically
4. Add secret: `RENDER_DEPLOY_HOOK_URL` (from Render dashboard → your service → Settings → Deploy Hook) to GitHub repository secrets
5. Every push to `main` now auto-deploys

> **Note:** Free tier spins down after 15 min of inactivity. First request after sleep takes ~30s to wake up.
