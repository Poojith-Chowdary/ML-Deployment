# --- path bootstrap ---
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ----------------------

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DATA_RAW,
    DROP_COLS,
    NUMERIC_FEATURES,
    TARGET_COL,
    TARGET_MAP,
)


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.drop(columns=DROP_COLS, errors="ignore")
    df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAP)
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor