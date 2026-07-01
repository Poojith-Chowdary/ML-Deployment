# --- path bootstrap ---
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ----------------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"

TARGET_COL = "Attrition"
TARGET_MAP = {"Yes": 1, "No": 0}
MODEL_VERSION = "1.0.0"

DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

NUMERIC_FEATURES = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

CATEGORICAL_FEATURES = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES