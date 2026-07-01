import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.config import ALL_FEATURES, NUMERIC_FEATURES
from src.preprocess import build_preprocessor, clean, load_raw_data


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data()


@pytest.fixture(scope="module")
def cleaned(raw_df):
    X, y = clean(raw_df)
    return X, y


def test_raw_data_loads(raw_df):
    assert len(raw_df) > 0
    assert "Attrition" in raw_df.columns


def test_raw_data_shape(raw_df):
    assert raw_df.shape[0] == 1470
    assert raw_df.shape[1] == 35


def test_clean_returns_correct_features(cleaned):
    X, y = cleaned
    assert list(X.columns) == ALL_FEATURES


def test_clean_no_nulls(cleaned):
    X, y = cleaned
    assert X.isnull().sum().sum() == 0


def test_target_is_binary(cleaned):
    X, y = cleaned
    assert set(y.unique()).issubset({0, 1})


def test_target_has_both_classes(cleaned):
    X, y = cleaned
    assert 0 in y.values and 1 in y.values


def test_drop_cols_removed(cleaned):
    X, _ = cleaned
    for col in ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]:
        assert col not in X.columns, f"Column {col} should have been dropped"


def test_feature_order_is_deterministic(raw_df):
    """Calling clean() twice should give same column order."""
    X1, _ = clean(raw_df)
    X2, _ = clean(raw_df)
    assert list(X1.columns) == list(X2.columns)


def test_preprocessor_output_has_no_nulls(cleaned):
    X, _ = cleaned
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert not np.isnan(transformed).any()


def test_preprocessor_output_shape(cleaned):
    X, _ = cleaned
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    # Numeric features + one-hot encoded categoricals — just check rows match
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > len(NUMERIC_FEATURES)  # OHE expands categoricals
