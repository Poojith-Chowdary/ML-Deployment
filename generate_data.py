"""
Run this once to generate the training dataset.
Usage: python generate_data.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n = 1470

age = np.random.randint(18, 60, n)
overtime = np.random.choice([0, 1], n, p=[0.72, 0.28])
job_sat = np.random.randint(1, 5, n)
work_life = np.random.randint(1, 5, n)
years_company = np.random.randint(0, 20, n)
monthly_income = np.random.randint(1000, 20000, n)
distance = np.random.randint(1, 30, n)
num_companies = np.random.randint(0, 9, n)
env_sat = np.random.randint(1, 5, n)

logit = (
    -1.2
    + 1.5  * overtime
    - 0.5  * job_sat
    - 0.4  * work_life
    - 0.04 * years_company
    - 0.00003 * monthly_income
    + 0.03 * distance
    + 0.2  * num_companies
    - 0.3  * env_sat
    + np.random.normal(0, 0.8, n)
)
prob = 1 / (1 + np.exp(-logit))
attrition = (np.random.uniform(0, 1, n) < prob).astype(int)

df = pd.DataFrame({
    "Age": age,
    "Attrition": np.where(attrition == 1, "Yes", "No"),
    "BusinessTravel": np.random.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], n, p=[0.71, 0.19, 0.10]),
    "DailyRate": np.random.randint(100, 1500, n),
    "Department": np.random.choice(["Sales", "Research & Development", "Human Resources"], n, p=[0.30, 0.65, 0.05]),
    "DistanceFromHome": distance,
    "Education": np.random.randint(1, 6, n),
    "EducationField": np.random.choice(["Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"], n),
    "EmployeeCount": 1,
    "EmployeeNumber": range(1, n + 1),
    "EnvironmentSatisfaction": env_sat,
    "Gender": np.random.choice(["Male", "Female"], n),
    "HourlyRate": np.random.randint(30, 100, n),
    "JobInvolvement": np.random.randint(1, 5, n),
    "JobLevel": np.random.randint(1, 6, n),
    "JobRole": np.random.choice(["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"], n),
    "JobSatisfaction": job_sat,
    "MaritalStatus": np.random.choice(["Single", "Married", "Divorced"], n),
    "MonthlyIncome": monthly_income,
    "MonthlyRate": np.random.randint(2000, 27000, n),
    "NumCompaniesWorked": num_companies,
    "Over18": "Y",
    "OverTime": np.where(overtime == 1, "Yes", "No"),
    "PercentSalaryHike": np.random.randint(11, 25, n),
    "PerformanceRating": np.random.choice([3, 4], n, p=[0.85, 0.15]),
    "RelationshipSatisfaction": np.random.randint(1, 5, n),
    "StandardHours": 80,
    "StockOptionLevel": np.random.randint(0, 4, n),
    "TotalWorkingYears": np.random.randint(0, 40, n),
    "TrainingTimesLastYear": np.random.randint(0, 7, n),
    "WorkLifeBalance": work_life,
    "YearsAtCompany": years_company,
    "YearsInCurrentRole": np.random.randint(0, 18, n),
    "YearsSinceLastPromotion": np.random.randint(0, 15, n),
    "YearsWithCurrManager": np.random.randint(0, 17, n),
})

out = Path("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Dataset saved to {out}")
print(f"Shape: {df.shape}")
print(f"Attrition rate: {df['Attrition'].value_counts().to_dict()}")