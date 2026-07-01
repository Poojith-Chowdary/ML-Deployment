import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import pytest
from src.config import MODEL_PATH, ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES


@pytest.fixture(scope="module")
def artifact():
    assert MODEL_PATH.exists(), f"model.joblib not found at {MODEL_PATH}. Run: python src/train.py"
    return joblib.load(MODEL_PATH)


def test_artifact_has_required_keys(artifact):
    assert "pipeline" in artifact
    assert "threshold" in artifact
    assert "version" in artifact


def test_threshold_is_valid(artifact):
    t = artifact["threshold"]
    assert isinstance(t, float)
    assert 0.0 < t < 1.0, f"Threshold {t} is outside (0, 1)"


def test_version_string(artifact):
    assert isinstance(artifact["version"], str)
    assert artifact["version"] == "1.0.0"


def test_predict_proba_shape(artifact, high_risk_row):
    proba = artifact["pipeline"].predict_proba(high_risk_row)
    assert proba.shape == (1, 2), f"Expected shape (1, 2), got {proba.shape}"


def test_predict_proba_sums_to_one(artifact, high_risk_row):
    proba = artifact["pipeline"].predict_proba(high_risk_row)
    total = proba[0].sum()
    assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}, not 1.0"


def test_probabilities_in_range(artifact, high_risk_row):
    proba = artifact["pipeline"].predict_proba(high_risk_row)
    assert 0.0 <= proba[0][1] <= 1.0


def test_feature_count(artifact, high_risk_row):
    # Pipeline should accept exactly our feature set without error
    artifact["pipeline"].predict_proba(high_risk_row)


def test_model_handles_batch(artifact):
    """Model should handle multiple rows at once."""
    rows = []
    for i in range(5):
        row = {f: 1 for f in NUMERIC_FEATURES}
        row.update({f: "Travel_Rarely" for f in ["BusinessTravel"]})
        row.update({f: "Sales" for f in ["Department"]})
        row.update({f: "Medical" for f in ["EducationField"]})
        row.update({f: "Male" for f in ["Gender"]})
        row.update({f: "Sales Executive" for f in ["JobRole"]})
        row.update({f: "Married" for f in ["MaritalStatus"]})
        row.update({f: "No" for f in ["OverTime"]})
        rows.append(row)
    df = pd.DataFrame(rows)[ALL_FEATURES]
    proba = artifact["pipeline"].predict_proba(df)
    assert proba.shape == (5, 2)


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def high_risk_row():
    data = {
        "Age": 28, "BusinessTravel": "Travel_Frequently", "DailyRate": 200,
        "Department": "Sales", "DistanceFromHome": 25, "Education": 2,
        "EducationField": "Marketing", "EnvironmentSatisfaction": 1,
        "Gender": "Male", "HourlyRate": 35, "JobInvolvement": 1,
        "JobLevel": 1, "JobRole": "Sales Representative", "JobSatisfaction": 1,
        "MaritalStatus": "Single", "MonthlyIncome": 1500, "MonthlyRate": 3000,
        "NumCompaniesWorked": 7, "OverTime": "Yes", "PercentSalaryHike": 11,
        "PerformanceRating": 3, "RelationshipSatisfaction": 1, "StockOptionLevel": 0,
        "TotalWorkingYears": 3, "TrainingTimesLastYear": 0, "WorkLifeBalance": 1,
        "YearsAtCompany": 1, "YearsInCurrentRole": 0, "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 0,
    }
    return pd.DataFrame([data])[ALL_FEATURES]
