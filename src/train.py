"""
==============================================================================
Module: src/train.py
Purpose: Synthetic Dataset Generation, Model Training, Evaluation & Artifact Export
==============================================================================

This script demonstrates the training stage of an MLOps pipeline:
1. Synthetic Data Generation: Deterministic, realistic house dataset.
2. Train / Test Split: Evaluates out-of-sample generalization.
3. Scikit-learn Pipeline: Combines StandardScaler + RandomForestRegressor.
   - Saves preprocessing transformations and estimator as a single artifact.
   - Prevents data leakage and ensures raw inputs can be served at inference.
4. Evaluation: Calculates MAE and RMSE (regression metrics).
5. Quality Gate: Asserts RMSE is below an acceptable threshold before exporting.
6. Artifact Export:
   - artifacts/house_price_pipeline.joblib (The trained pipeline)
   - artifacts/reference_data.csv (Baseline feature distributions for monitoring)
   - artifacts/metrics.json (Evaluation summary for auditability and CI)
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")

# Paths and configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Configurable Quality Gate Threshold (Can be overridden by environment variable)
MAX_ALLOWED_RMSE = float(os.getenv("MAX_ALLOWED_RMSE", "35000.0"))
RANDOM_SEED = 42


def generate_synthetic_data(n_samples: int = 1500, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generates a reproducible synthetic dataset for house price prediction.

    Features:
    - area_sqft: Total floor area (clipped between 600 and 4500 sq ft)
    - bedrooms: Number of bedrooms (discrete 1 to 5)
    - age_years: Age of the property (0 to 50 years)

    Target:
    - price: Base price + (area * 180) + (bedrooms * 20,000) - (age * 1,200) + Gaussian noise
    """
    logger.info(f"Generating {n_samples} synthetic housing records with seed={seed}...")
    rng = np.random.default_rng(seed)

    area_sqft = rng.normal(loc=2000, scale=600, size=n_samples)
    area_sqft = np.clip(area_sqft, 600, 4500).round(1)

    bedrooms = rng.integers(low=1, high=6, size=n_samples)
    age_years = rng.uniform(low=0, high=50, size=n_samples).round(1)

    # Realistic price generation formula
    # Base: $50,000
    # Area: $180 / sqft
    # Bedrooms: $20,000 / bedroom
    # Age depreciation: -$1,200 / year
    # Small noise: Gaussian std = $15,000
    noise = rng.normal(loc=0, scale=15000, size=n_samples)

    price = (
        50000.0
        + (area_sqft * 180.0)
        + (bedrooms * 20000.0)
        - (age_years * 1200.0)
        + noise
    )
    # Ensure all prices remain positive
    price = np.clip(price, 50000.0, None).round(2)

    df = pd.DataFrame({
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "age_years": age_years,
        "price": price,
    })

    logger.info(f"Dataset generated. Sample stats:\n{df.describe().round(2)}")
    return df


def build_pipeline() -> Pipeline:
    """
    Constructs an end-to-end Scikit-learn Pipeline.

    Why a Pipeline is crucial in MLOps:
    - Preprocessing (StandardScaler) is coupled directly with the estimator (RandomForestRegressor).
    - When exported, the pipeline accepts raw features (area_sqft, bedrooms, age_years)
      and handles standard scaling internally at inference time.
    - Prevents data leakage between train and test sets.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=12,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_evaluate():
    """
    Executes the training and evaluation workflow, enforcing validation gates.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic data
    df = generate_synthetic_data()

    feature_cols = ["area_sqft", "bedrooms", "age_years"]
    target_col = "price"

    X = df[feature_cols]
    y = df[target_col]

    # 2. Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    logger.info(f"Split completed: {len(X_train)} train samples, {len(X_test)} test samples.")

    # 3. Build & Fit Pipeline
    pipeline = build_pipeline()
    logger.info("Fitting Scikit-learn pipeline (StandardScaler + RandomForestRegressor)...")
    pipeline.fit(X_train, y_train)

    # 4. Evaluate on Test Set
    logger.info("Evaluating fitted model against out-of-sample test set...")
    y_pred = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))

    logger.info(f"Evaluation Results -> MAE: ${mae:,.2f} | RMSE: ${rmse:,.2f}")
    logger.info(f"Configured Quality Gate Threshold -> MAX RMSE: ${MAX_ALLOWED_RMSE:,.2f}")

    # 5. Model Validation Gate
    # In production CI/CD, bad models must be rejected before deployment!
    if rmse > MAX_ALLOWED_RMSE:
        error_msg = (
            f"MODEL VALIDATION FAILED: Test RMSE (${rmse:,.2f}) exceeded "
            f"acceptable threshold (${MAX_ALLOWED_RMSE:,.2f})."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("SUCCESS: Model passed quality gate threshold!")

    # 6. Save Artifacts
    # Artifact A: Serialized Pipeline
    model_path = ARTIFACTS_DIR / "house_price_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    logger.info(f"Saved model pipeline to: {model_path}")

    # Artifact B: Reference Training Data (used for baseline data-drift monitoring)
    reference_data_path = ARTIFACTS_DIR / "reference_data.csv"
    X_train.to_csv(reference_data_path, index=False)
    logger.info(f"Saved reference feature data to: {reference_data_path}")

    # Artifact C: Metrics Metadata
    metrics = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "threshold_rmse": MAX_ALLOWED_RMSE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_seed": RANDOM_SEED,
        "model_type": "Pipeline(StandardScaler, RandomForestRegressor)",
    }
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics metadata to: {metrics_path}")

    return metrics


if __name__ == "__main__":
    try:
        train_and_evaluate()
        logger.info("Training pipeline finished successfully.")
    except Exception as e:
        logger.exception(f"Training pipeline terminated with error: {e}")
        sys.exit(1)

