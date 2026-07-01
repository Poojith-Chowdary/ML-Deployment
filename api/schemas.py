from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    Age: int = Field(..., ge=18, le=65, description="Employee age")
    BusinessTravel: Literal["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    DailyRate: int = Field(..., ge=100, le=1500)
    Department: Literal["Sales", "Research & Development", "Human Resources"]
    DistanceFromHome: int = Field(..., ge=1, le=30)
    Education: int = Field(..., ge=1, le=5)
    EducationField: Literal["Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"]
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    Gender: Literal["Male", "Female"]
    HourlyRate: int = Field(..., ge=30, le=100)
    JobInvolvement: int = Field(..., ge=1, le=4)
    JobLevel: int = Field(..., ge=1, le=5)
    JobRole: Literal[
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ]
    JobSatisfaction: int = Field(..., ge=1, le=4)
    MaritalStatus: Literal["Single", "Married", "Divorced"]
    MonthlyIncome: int = Field(..., ge=1000, le=20000)
    MonthlyRate: int = Field(..., ge=2000, le=27000)
    NumCompaniesWorked: int = Field(..., ge=0, le=9)
    OverTime: Literal["Yes", "No"]
    PercentSalaryHike: int = Field(..., ge=11, le=25)
    PerformanceRating: int = Field(..., ge=3, le=4)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    StockOptionLevel: int = Field(..., ge=0, le=3)
    TotalWorkingYears: int = Field(..., ge=0, le=40)
    TrainingTimesLastYear: int = Field(..., ge=0, le=6)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    YearsAtCompany: int = Field(..., ge=0, le=40)
    YearsInCurrentRole: int = Field(..., ge=0, le=18)
    YearsSinceLastPromotion: int = Field(..., ge=0, le=15)
    YearsWithCurrManager: int = Field(..., ge=0, le=17)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 35,
                "BusinessTravel": "Travel_Frequently",
                "DailyRate": 500,
                "Department": "Sales",
                "DistanceFromHome": 20,
                "Education": 3,
                "EducationField": "Marketing",
                "EnvironmentSatisfaction": 2,
                "Gender": "Male",
                "HourlyRate": 45,
                "JobInvolvement": 2,
                "JobLevel": 2,
                "JobRole": "Sales Executive",
                "JobSatisfaction": 1,
                "MaritalStatus": "Single",
                "MonthlyIncome": 3500,
                "MonthlyRate": 8000,
                "NumCompaniesWorked": 5,
                "OverTime": "Yes",
                "PercentSalaryHike": 12,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 2,
                "StockOptionLevel": 0,
                "TotalWorkingYears": 8,
                "TrainingTimesLastYear": 1,
                "WorkLifeBalance": 1,
                "YearsAtCompany": 2,
                "YearsInCurrentRole": 1,
                "YearsSinceLastPromotion": 1,
                "YearsWithCurrManager": 1
            }
        }
    }


class PredictResponse(BaseModel):
    prediction: Literal["Churn", "Retain"]
    probability: float = Field(..., description="Probability of churning (0–1)")
    model_version: str
    threshold_used: float = Field(..., description="Decision threshold applied")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    decision_threshold: float
    feature_count: int
    numeric_features: list[str]
    categorical_features: list[str]
