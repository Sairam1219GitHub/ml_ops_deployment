"""
==============================================================================
Module: src/monitor.py
Purpose: Data Drift Monitoring & Telemetry Analysis Demonstration
==============================================================================

This module demonstrates how deployed ML systems detect feature shifts
between the reference training baseline and live production inference requests.

MLOPS MONITORING TAXONOMY:
-------------------------
A comprehensive production MLOps system monitors three distinct layers:

1. API & Infrastructure Health:
   - Tracks CPU, memory, HTTP request rates (QPS), p50/p95/p99 latency, 4xx/5xx errors.
   - Tells us: "Is the server alive and serving requests quickly?"

2. Data Drift / Feature Shift Monitoring (Implemented in this script):
   - Compares incoming features P(X_live) against baseline training features P(X_ref).
   - Early warning indicator: warns engineers that production input data has changed
     before actual ground truth labels are even available!
   - Tells us: "Are users asking for predictions on the kind of properties we trained on?"

3. Model Quality / Performance Monitoring:
   - Calculates true error metrics (MAE, RMSE, R2, accuracy).
   - REQUIRES GROUND TRUTH: In real estate, the actual sold price is only known
     weeks or months later (the "ground truth feedback delay").
   - Once true prices arrive, engineers join predictions with outcomes to evaluate decay.

4. Concept Drift:
   - When the underlying relationship P(Y | X) changes (e.g., inflation or market crash
     causes a 2,000 sqft house that sold for $400k last year to sell for $600k this year).

IMPORTANT TEACHING NOTE ON THIS SCRIPT:
--------------------------------------
This script compares statistical summaries (mean, min, max, count) and flags
a shift if the relative difference between reference mean and recent mean
exceeds a configurable threshold (e.g. 20%).
In industrial production, engineers use statistically rigorous distribution
tests such as:
- Kolmogorov-Smirnov (KS) test or Mann-Whitney U test (continuous features)
- Population Stability Index (PSI)
- Wasserstein Distance (Earth Mover's Distance)
- Tools such as Evidently AI, Great Expectations, or Whylogs.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("monitor")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REFERENCE_DATA_PATH = ARTIFACTS_DIR / "reference_data.csv"
PREDICTION_LOGS_PATH = ARTIFACTS_DIR / "prediction_logs.csv"
MONITORING_REPORT_PATH = ARTIFACTS_DIR / "monitoring_report.json"

# Configurable threshold: 0.20 = 20% relative shift in mean flags an alert
DRIFT_THRESHOLD_RATIO = float(os.getenv("DRIFT_THRESHOLD_RATIO", "0.20"))


def compute_column_statistics(series: pd.Series) -> Dict[str, float]:
    """
    Computes count, mean, std, min, max for a numerical series.
    """
    clean_series = series.dropna()
    if clean_series.empty:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    return {
        "count": int(clean_series.count()),
        "mean": round(float(clean_series.mean()), 2),
        "std": round(float(clean_series.std()), 2),
        "min": round(float(clean_series.min()), 2),
        "max": round(float(clean_series.max()), 2),
    }


def analyze_drift(reference_df: pd.DataFrame, recent_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compares recent inference feature distributions with reference training features.
    """
    feature_columns = ["area_sqft", "bedrooms", "age_years"]
    features_report = {}
    any_drift_detected = False

    for col in feature_columns:
        if col not in reference_df.columns or col not in recent_df.columns:
            logger.warning(f"Feature '{col}' missing from one of the datasets. Skipping.")
            continue

        ref_stats = compute_column_statistics(reference_df[col])
        rec_stats = compute_column_statistics(recent_df[col])

        # Calculate relative difference in mean
        # Avoid division by zero
        ref_mean = ref_stats["mean"]
        rec_mean = rec_stats["mean"]

        if abs(ref_mean) > 1e-6:
            relative_diff = abs(rec_mean - ref_mean) / abs(ref_mean)
        else:
            relative_diff = abs(rec_mean - ref_mean)

        is_drifted = bool(relative_diff > DRIFT_THRESHOLD_RATIO)
        if is_drifted:
            any_drift_detected = True

        features_report[col] = {
            "reference_statistics": ref_stats,
            "recent_statistics": rec_stats,
            "relative_mean_difference": round(float(relative_diff), 4),
            "threshold_applied": DRIFT_THRESHOLD_RATIO,
            "drift_detected": is_drifted,
            "status": "ALERT_DRIFT_DETECTED" if is_drifted else "STABLE",
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_drift_detected": any_drift_detected,
        "sample_sizes": {
            "reference_samples": int(len(reference_df)),
            "recent_logged_samples": int(len(recent_df)),
        },
        "threshold_ratio": DRIFT_THRESHOLD_RATIO,
        "feature_metrics": features_report,
        "teaching_note": (
            "This report demonstrates statistical shift detection between baseline training data "
            "and live inference telemetry. Model quality monitoring (MAE/RMSE on production traffic) "
            "requires true sale prices, which are subject to real-world ground-truth delay."
        ),
    }


def generate_simulated_logs(drifted: bool = False, n_records: int = 50) -> pd.DataFrame:
    """
    Helper for classroom demos: synthesizes realistic or drifted prediction logs
    if the student has not yet executed sufficient API requests.
    """
    logger.info(f"Generating {n_records} simulated inference logs (drifted={drifted})...")
    rng = np.random.default_rng(99)

    if drifted:
        # Simulate luxury mansions (significantly larger area and more bedrooms)
        area_sqft = rng.normal(loc=3800, scale=400, size=n_records).clip(2500, 6000).round(1)
        bedrooms = rng.integers(low=4, high=8, size=n_records)
        age_years = rng.uniform(low=0, high=10, size=n_records).round(1)
    else:
        # Simulate typical properties matching training distributions
        area_sqft = rng.normal(loc=2000, scale=500, size=n_records).clip(800, 4000).round(1)
        bedrooms = rng.integers(low=1, high=6, size=n_records)
        age_years = rng.uniform(low=5, high=40, size=n_records).round(1)

    predicted_price = (50000 + area_sqft * 180 + bedrooms * 20000 - age_years * 1200).round(2)

    return pd.DataFrame({
        "timestamp": [datetime.now(timezone.utc).isoformat()] * n_records,
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "age_years": age_years,
        "predicted_price": predicted_price,
    })


def run_monitoring(simulate_if_empty: bool = True):
    """
    Loads baseline reference data and recent prediction logs, computes drift,
    and writes artifacts/monitoring_report.json.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not REFERENCE_DATA_PATH.exists():
        error_msg = (
            f"Reference data missing at {REFERENCE_DATA_PATH}. "
            f"Please run 'python -m src.train' first."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    reference_df = pd.read_csv(REFERENCE_DATA_PATH)
    logger.info(f"Loaded reference training data with {len(reference_df)} rows.")

    # Check for live logs
    if PREDICTION_LOGS_PATH.exists() and os.path.getsize(PREDICTION_LOGS_PATH) > 0:
        recent_df = pd.read_csv(PREDICTION_LOGS_PATH)
        logger.info(f"Loaded live prediction logs with {len(recent_df)} rows.")
    elif simulate_if_empty:
        logger.warning(
            f"No live prediction logs found at {PREDICTION_LOGS_PATH}. "
            f"Generating simulated demonstration traffic..."
        )
        recent_df = generate_simulated_logs(drifted=False, n_records=50)
        recent_df.to_csv(PREDICTION_LOGS_PATH, index=False)
    else:
        logger.error(f"No prediction logs found at {PREDICTION_LOGS_PATH}.")
        sys.exit(1)

    # Perform drift analysis
    report = analyze_drift(reference_df=reference_df, recent_df=recent_df)

    # Write report artifact
    with open(MONITORING_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Monitoring report saved to: {MONITORING_REPORT_PATH}")
    logger.info(
        f"Monitoring Result -> Overall Drift Detected: {report['overall_drift_detected']}"
    )

    for feature, details in report["feature_metrics"].items():
        ref_mean = details["reference_statistics"]["mean"]
        rec_mean = details["recent_statistics"]["mean"]
        diff_pct = details["relative_mean_difference"] * 100
        status_label = details["status"]
        logger.info(
            f" - Feature '{feature}': Ref Mean={ref_mean} | Live Mean={rec_mean} | "
            f"Diff={diff_pct:.1f}% -> {status_label}"
        )

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MLOps feature drift monitoring.")
    parser.add_argument(
        "--simulate-drift",
        action="store_true",
        help="Simulate a batch of drifted incoming requests for demonstration.",
    )
    args = parser.parse_args()

    if args.simulate_drift:
        drifted_df = generate_simulated_logs(drifted=True, n_records=60)
        drifted_df.to_csv(PREDICTION_LOGS_PATH, index=False)
        logger.info("Injected drifted requests into prediction_logs.csv.")

    run_monitoring()

