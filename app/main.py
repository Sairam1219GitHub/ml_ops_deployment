"""
==============================================================================
Module: app/main.py
Purpose: FastAPI Inference Microservice for House Price Prediction
==============================================================================

Key Responsibilities:
1. Lifespan Model Management: Loads the serialized Scikit-learn Pipeline
   (StandardScaler + RandomForestRegressor) into memory upon startup.
2. Request Validation: Uses Pydantic to ensure input bounds and types.
3. Prediction Inference: Transforms raw features and generates price estimation.
4. Telemetry / Logging: Appends incoming request features and model predictions
   to artifacts/prediction_logs.csv for continuous monitoring demonstrations.

IMPORTANT TEACHING NOTE ON CSV LOGGING:
--------------------------------------
In this classroom repository, predictions are appended to a local CSV file
('artifacts/prediction_logs.csv') so students can easily observe incoming
production traffic and compare it against the training reference data.

IN A REAL PRODUCTION ENVIRONMENT:
- Local file logging fails in multi-worker containers (concurrency race conditions).
- Ephemeral containers (like AWS Fargate or Google Cloud Run) lose local disk state on restart.
- Production architectures route telemetry asynchronously to managed streaming
  or data-lake infrastructure (e.g., Google Cloud Pub/Sub -> BigQuery,
  AWS Kinesis -> S3 / Snowflake, or specialized ML observability platforms like
  Evidently AI, Arize, or Fiddler) while strictly complying with PII/GDPR privacy rules.
"""

import csv
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

# Directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "house_price_pipeline.joblib"
PREDICTION_LOGS_PATH = ARTIFACTS_DIR / "prediction_logs.csv"

# Global state for loaded model pipeline
model_pipeline = None


# ============================================================================
# Pydantic Schemas for Validation and OpenAPI Documentation
# ============================================================================
class HousePriceRequest(BaseModel):
    """
    Input schema defining the features required for predicting house price.
    Includes validation constraints and example values for Swagger docs.
    """
    area_sqft: float = Field(
        ...,
        gt=0.0,
        le=20000.0,
        description="Gross floor area of the house in square feet (must be > 0 and <= 20,000).",
        examples=[2150.0],
    )
    bedrooms: int = Field(
        ...,
        ge=1,
        le=20,
        description="Number of bedrooms (integer between 1 and 20).",
        examples=[3],
    )
    age_years: float = Field(
        ...,
        ge=0.0,
        le=200.0,
        description="Age of the house in years since construction (>= 0).",
        examples=[12.5],
    )


class HousePriceResponse(BaseModel):
    """
    Output schema returned to the client upon successful prediction.
    """
    predicted_price: float = Field(
        ...,
        description="Predicted market price of the house in USD.",
        examples=[452300.50],
    )
    currency: str = Field(
        default="USD",
        description="Currency denomination.",
    )
    features: Dict[str, Any] = Field(
        ...,
        description="Echo of the input features received by the model.",
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of inference execution.",
    )


class HealthResponse(BaseModel):
    """
    Health check response schema.
    """
    status: str
    model_loaded: bool
    version: str
    artifacts_path: str


# ============================================================================
# Helper Functions
# ============================================================================
def log_prediction_to_csv(features: Dict[str, Any], predicted_price: float, timestamp_iso: str):
    """
    Appends the inference record to a local CSV file for the drift monitoring demo.
    Creates header row if the file does not yet exist.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = PREDICTION_LOGS_PATH.is_file()

    fieldnames = ["timestamp", "area_sqft", "bedrooms", "age_years", "predicted_price"]

    try:
        with open(PREDICTION_LOGS_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "timestamp": timestamp_iso,
                "area_sqft": features["area_sqft"],
                "bedrooms": features["bedrooms"],
                "age_years": features["age_years"],
                "predicted_price": round(predicted_price, 2),
            })
    except Exception as e:
        logger.warning(f"Failed to append prediction to log CSV: {e}")


# ============================================================================
# Application Lifespan: Model Loading on Startup
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the trained model pipeline into memory before accepting requests,
    and cleans up resources upon shutdown.
    """
    global model_pipeline
    logger.info("Initializing FastAPI inference service...")

    if not MODEL_PATH.exists():
        logger.error(f"Model artifact not found at {MODEL_PATH}!")
        logger.error("Please run 'python -m src.train' to train and export the pipeline first.")
        model_pipeline = None
    else:
        try:
            logger.info(f"Loading Scikit-learn pipeline from {MODEL_PATH}...")
            model_pipeline = joblib.load(MODEL_PATH)
            logger.info("Model pipeline loaded successfully into memory.")
        except Exception as e:
            logger.exception(f"Failed to load model pipeline: {e}")
            model_pipeline = None

    yield

    logger.info("Shutting down FastAPI inference service...")


# ============================================================================
# FastAPI App Instance
# ============================================================================
app = FastAPI(
    title="House Price Prediction API",
    description=(
        "Production-style MLOps prediction service using FastAPI, Scikit-learn Pipeline, "
        "and automated telemetry logging."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/", tags=["General"])
def read_root():
    """
    Service landing endpoint with useful quick links.
    """
    return {
        "service": "House Price Prediction API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "predict_endpoint": "/predict",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Monitoring"],
)
def check_health():
    """
    Liveness and readiness probe for container orchestrators (e.g. Kubernetes, Cloud Run).
    Returns 200 if the service is running and reports whether the model artifact is loaded.
    """
    is_ready = model_pipeline is not None
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        version="1.0.0",
        artifacts_path=str(MODEL_PATH),
    )


@app.post(
    "/predict",
    response_model=HousePriceResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
def predict_house_price(request: HousePriceRequest):
    """
    Accepts property characteristics and returns the predicted market price.

    Validation rules:
    - area_sqft: > 0 and <= 20,000
    - bedrooms: 1 to 20
    - age_years: 0 to 200
    """
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model pipeline is not loaded on this server. "
                "Ensure artifacts/house_price_pipeline.joblib exists and restart the service."
            ),
        )

    # 1. Format input features into a single-row DataFrame matching training column names
    input_df = pd.DataFrame([request.model_dump()])

    # 2. Execute inference through the Scikit-learn pipeline (applies StandardScaler -> RandomForest)
    try:
        raw_prediction = model_pipeline.predict(input_df)[0]
        prediction_val = float(round(raw_prediction, 2))
    except Exception as e:
        logger.exception(f"Inference error during model.predict(): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference calculation error: {str(e)}",
        )

    now_iso = datetime.now(timezone.utc).isoformat()

    # 3. Log the request features and prediction for continuous drift monitoring demonstration
    log_prediction_to_csv(
        features=request.model_dump(),
        predicted_price=prediction_val,
        timestamp_iso=now_iso,
    )

    return HousePriceResponse(
        predicted_price=prediction_val,
        currency="USD",
        features=request.model_dump(),
        timestamp=now_iso,
    )

