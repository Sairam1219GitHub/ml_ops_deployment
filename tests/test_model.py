"""
==============================================================================
Module: tests/test_model.py
Purpose: Unit Tests for Serialized Scikit-learn Model Pipeline
==============================================================================

Verifies:
1. Pipeline artifact existence and successful deserialization.
2. Prediction execution with valid single-row and batch DataFrames.
3. Prediction value validity (positive floating point numbers).
4. Directional sanity check (e.g., larger square footage produces a higher price).
5. Metrics validation against acceptable training thresholds.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "house_price_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


@pytest.fixture(scope="module")
def loaded_pipeline():
    """
    Loads the trained model pipeline fixture.
    If the artifact is missing, triggers training once to ensure test repeatability.
    """
    if not MODEL_PATH.exists():
        from src.train import train_and_evaluate
        train_and_evaluate()

    assert MODEL_PATH.exists(), f"Model pipeline artifact missing at {MODEL_PATH}"
    pipeline = joblib.load(MODEL_PATH)
    return pipeline


def test_model_artifact_exists_and_loads(loaded_pipeline):
    """
    Ensures the pipeline can be deserialized and contains expected pipeline steps.
    """
    assert loaded_pipeline is not None
    assert hasattr(loaded_pipeline, "predict")
    # Verify the pipeline includes both scaler and regressor steps
    steps = [name for name, _ in loaded_pipeline.steps]
    assert "scaler" in steps
    assert "regressor" in steps


def test_model_prediction_single_sample(loaded_pipeline):
    """
    Validates inference on a single sample property.
    """
    sample_data = pd.DataFrame([{
        "area_sqft": 2000.0,
        "bedrooms": 3,
        "age_years": 10.0,
    }])

    prediction = loaded_pipeline.predict(sample_data)

    assert len(prediction) == 1
    pred_val = float(prediction[0])
    assert not np.isnan(pred_val), "Prediction should not be NaN"
    assert pred_val > 0.0, f"Predicted price must be positive, got {pred_val}"
    # Realistic bounds check: a 2,000 sq ft 3-bedroom house should reasonably cost between $100k and $1M
    assert 100000.0 <= pred_val <= 1000000.0, f"Unrealistic prediction value: ${pred_val:,.2f}"


def test_model_directional_sanity(loaded_pipeline):
    """
    Behavioral Test: All else equal, a 3,500 sq ft home must be valued higher
    than a 1,200 sq ft home.
    """
    small_house = pd.DataFrame([{"area_sqft": 1200.0, "bedrooms": 3, "age_years": 5.0}])
    large_house = pd.DataFrame([{"area_sqft": 3500.0, "bedrooms": 3, "age_years": 5.0}])

    pred_small = float(loaded_pipeline.predict(small_house)[0])
    pred_large = float(loaded_pipeline.predict(large_house)[0])

    assert pred_large > pred_small, (
        f"Directional test failed: Large house (${pred_large:,.2f}) should cost more "
        f"than small house (${pred_small:,.2f})"
    )


def test_evaluation_metrics_artifact():
    """
    Verifies that metrics.json exists and RMSE meets the documented threshold.
    """
    assert METRICS_PATH.exists(), f"Metrics artifact missing at {METRICS_PATH}"
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "threshold_rmse" in metrics

    assert metrics["rmse"] <= metrics["threshold_rmse"], (
        f"Model RMSE ({metrics['rmse']}) violates threshold ({metrics['threshold_rmse']})"
    )

