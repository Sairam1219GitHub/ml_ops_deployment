"""
==============================================================================
Module: tests/test_api.py
Purpose: Integration Tests for FastAPI Prediction Microservice
==============================================================================

Uses FastAPI TestClient (HTTPX backed) to verify:
1. GET /health returns 200 and reports healthy with model loaded.
2. POST /predict accepts valid payloads and returns 200 with valid positive prices.
3. POST /predict rejects invalid payloads (e.g., negative area, zero bedrooms) with 422.
4. Telemetry logging correctly records requests to prediction_logs.csv.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import PREDICTION_LOGS_PATH, app


@pytest.fixture(scope="module")
def client():
    """
    Creates a FastAPI TestClient wrapped in the app lifespan context.
    """
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """
    GET /health should return HTTP 200 with model_loaded == True.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data


def test_predict_endpoint_valid_payload(client):
    """
    POST /predict with valid input features returns HTTP 200 and a realistic positive price.
    """
    payload = {
        "area_sqft": 2200.0,
        "bedrooms": 3,
        "age_years": 8.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "predicted_price" in data
    assert "features" in data
    assert "timestamp" in data

    predicted_price = data["predicted_price"]
    assert isinstance(predicted_price, (int, float))
    assert predicted_price > 0.0
    # Sanity bounds check
    assert 100000.0 <= predicted_price <= 1500000.0

    # Ensure echo features match input
    assert data["features"]["area_sqft"] == 2200.0
    assert data["features"]["bedrooms"] == 3
    assert data["features"]["age_years"] == 8.0


def test_predict_validation_errors(client):
    """
    Pydantic schema validation should reject invalid input with HTTP 422 Unprocessable Entity.
    """
    # Negative area
    invalid_area = {"area_sqft": -500.0, "bedrooms": 3, "age_years": 10.0}
    res = client.post("/predict", json=invalid_area)
    assert res.status_code == 422

    # Zero bedrooms
    invalid_bedrooms = {"area_sqft": 2000.0, "bedrooms": 0, "age_years": 10.0}
    res = client.post("/predict", json=invalid_bedrooms)
    assert res.status_code == 422

    # Negative age
    invalid_age = {"area_sqft": 2000.0, "bedrooms": 3, "age_years": -5.0}
    res = client.post("/predict", json=invalid_age)
    assert res.status_code == 422


def test_prediction_telemetry_logging(client):
    """
    Verifies that successful predictions append an entry to prediction_logs.csv.
    """
    payload = {
        "area_sqft": 1750.0,
        "bedrooms": 2,
        "age_years": 15.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    assert PREDICTION_LOGS_PATH.exists(), "Prediction log file was not created"
    with open(PREDICTION_LOGS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
        assert "1750" in content
        assert "timestamp" in content

