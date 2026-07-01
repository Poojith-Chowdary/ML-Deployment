import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="session")
def client():
    """Single TestClient for the whole test session — model loads once."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def high_risk_payload():
    """Employee profile with many churn risk factors."""
    return {
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
        "YearsWithCurrManager": 0,
    }


@pytest.fixture
def low_risk_payload(high_risk_payload):
    """Employee profile with few churn risk factors."""
    return {
        **high_risk_payload,
        "OverTime": "No",
        "JobSatisfaction": 4,
        "WorkLifeBalance": 4,
        "EnvironmentSatisfaction": 4,
        "MonthlyIncome": 15000,
        "NumCompaniesWorked": 1,
        "YearsAtCompany": 10,
        "DistanceFromHome": 2,
    }
