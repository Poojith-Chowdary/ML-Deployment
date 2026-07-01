import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



# ── /health ─────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_model_loaded(client):
    r = client.get("/health")
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


# ── /model-info ──────────────────────────────────────────────────────────────

def test_model_info_returns_200(client):
    r = client.get("/model-info")
    assert r.status_code == 200


def test_model_info_fields(client):
    r = client.get("/model-info")
    data = r.json()
    assert "model_version" in data
    assert "decision_threshold" in data
    assert "feature_count" in data
    assert data["feature_count"] == 30


def test_model_info_threshold_valid(client):
    r = client.get("/model-info")
    t = r.json()["decision_threshold"]
    assert 0.0 < t < 1.0


# ── POST /predict — valid payloads ──────────────────────────────────────────

def test_predict_high_risk_returns_200(client, high_risk_payload):
    r = client.post("/predict", json=high_risk_payload)
    assert r.status_code == 200


def test_predict_response_has_required_fields(client, high_risk_payload):
    r = client.post("/predict", json=high_risk_payload)
    data = r.json()
    assert "prediction" in data
    assert "probability" in data
    assert "model_version" in data
    assert "threshold_used" in data


def test_predict_prediction_is_valid_label(client, high_risk_payload):
    r = client.post("/predict", json=high_risk_payload)
    assert r.json()["prediction"] in ("Churn", "Retain")


def test_predict_probability_in_range(client, high_risk_payload):
    r = client.post("/predict", json=high_risk_payload)
    prob = r.json()["probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_high_risk_is_churn(client, high_risk_payload):
    """High-risk profile should predict Churn."""
    r = client.post("/predict", json=high_risk_payload)
    assert r.json()["prediction"] == "Churn"


def test_predict_low_risk_is_retain(client, low_risk_payload):
    """Low-risk profile should predict Retain."""
    r = client.post("/predict", json=low_risk_payload)
    assert r.json()["prediction"] == "Retain"


def test_predict_model_version(client, high_risk_payload):
    r = client.post("/predict", json=high_risk_payload)
    assert r.json()["model_version"] == "1.0.0"


# ── POST /predict — invalid payloads ────────────────────────────────────────

def test_predict_missing_all_fields_returns_422(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_missing_one_field_returns_422(client, high_risk_payload):
    payload = {k: v for k, v in high_risk_payload.items() if k != "OverTime"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_wrong_type_returns_422(client, high_risk_payload):
    payload = {**high_risk_payload, "Age": "not-a-number"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_enum_returns_422(client, high_risk_payload):
    payload = {**high_risk_payload, "OverTime": "Maybe"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_age_out_of_range_returns_422(client, high_risk_payload):
    payload = {**high_risk_payload, "Age": 200}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_empty_body_returns_422(client):
    r = client.post("/predict")
    assert r.status_code == 422
