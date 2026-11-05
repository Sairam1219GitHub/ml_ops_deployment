# End-to-End MLOps Demonstration: House Price Prediction

> **A beginner-friendly, production-style MLOps practical guide** demonstrating the full machine learning lifecycle:  
> **Data & Feature Engineering &rarr; Preprocessing Pipeline &rarr; Model Training & Evaluation &rarr; Quality Gates &rarr; Model Artifact Serialization &rarr; FastAPI Microservice &rarr; Docker Containerization &rarr; GitHub Actions CI/CD &rarr; Drift Monitoring &rarr; Retraining Loop (CT)**.

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Project Directory Structure](#2-project-directory-structure)
3. [Prerequisites & Local Installation](#3-prerequisites--local-installation)
4. [Step-by-Step Hands-On Walkthrough](#4-step-by-step-hands-on-walkthrough)
   - [Step 1: Train the Model & Export Artifacts](#step-1-train-the-model--export-artifacts)
   - [Step 2: Inspect Generated Artifacts](#step-2-inspect-generated-artifacts)
   - [Step 3: Run the FastAPI Microservice](#step-3-run-the-fastapi-microservice)
   - [Step 4: Test Endpoints via Swagger UI & cURL](#step-4-test-endpoints-via-swagger-ui--curl)
   - [Step 5: Run the Automated Pytest Suite](#step-5-run-the-automated-pytest-suite)
   - [Step 6: Run Data Drift Monitoring](#step-6-run-data-drift-monitoring)
   - [Step 7: Build & Run with Docker](#step-7-build--run-with-docker)
5. [CI/CD Workflows (GitHub Actions)](#5-cicd-workflows-github-actions)
   - [Continuous Integration (CI)](#continuous-integration-ci)
   - [Continuous Deployment (CD) to Google Cloud Run](#continuous-deployment-cd-to-google-cloud-run)
   - [Scheduled Drift Monitoring Workflow](#scheduled-drift-monitoring-workflow)
6. [Core MLOps Concepts Explained](#6-core-mlops-concepts-explained)
   - [CI vs. CD vs. Continuous Training (CT)](#ci-vs-cd-vs-continuous-training-ct)
   - [The Three Layers of MLOps Monitoring](#the-three-layers-of-mlops-monitoring)
   - [The Ground-Truth Feedback Delay](#the-ground-truth-feedback-delay)
   - [Why Pipelines & Single Artifacts Matter](#why-pipelines--single-artifacts-matter)
   - [Model Governance & Safe Promotion](#model-governance--safe-promotion)

---

## 1. Architecture Overview

```
                      +-----------------------------+
                      |   Synthetic Housing Data    |
                      | (area_sqft, bedrooms, age)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Train / Test Split (80/20) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Scikit-Learn Pipeline     |
                      | StandardScaler + RFRegressor|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Evaluation & Quality     |
                      |     Gate (RMSE < $35k)      |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |       Model Artifacts       |
                      |   .joblib, .csv, metrics    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    FastAPI Microservice     |
                      |    /health and /predict     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |      Docker Container       |
                      |    python:3.11-slim base    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Google Cloud Run / Cloud  |
                      |     Serverless Hosting      |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Telemetry & Drift Monitor   |
                      | P(X_live) vs. P(X_ref)      |
                      +--------------+--------------+
                                     |
                      Shift / Performance Decay
                                     |
                                     v
                      +-----------------------------+
                      | Retrain, Validate & Promote |
                      |    (Continuous Training)    |
                      +-----------------------------+
```

---

## 2. Project Directory Structure

```
mlops_class/
|-- app/
|   |-- __init__.py            # FastAPI package marker
|   `-- main.py                # FastAPI app with lifespan loader, /health, /predict & telemetry
|-- src/
|   |-- __init__.py            # Source package marker
|   |-- train.py               # Synthetic data, Pipeline training, MAE/RMSE eval & artifact export
|   `-- monitor.py             # Feature distribution summary, drift detection & report export
|-- tests/
|   |-- __init__.py            # Test package marker
|   |-- test_model.py          # Unit tests: pipeline loading, prediction sanity, metrics checks
|   `-- test_api.py            # Integration tests: /health, /predict validation, CSV telemetry
|-- artifacts/                 # Directory holding serialised assets (generated on training)
|   |-- .gitkeep               # Keeps folder tracked in Git
|   |-- house_price_pipeline.joblib   # Serialized StandardScaler + RandomForest pipeline
|   |-- reference_data.csv            # Training feature baseline for drift monitoring
|   |-- metrics.json                  # Evaluated MAE, RMSE and training metadata
|   |-- prediction_logs.csv           # Runtime request logs (generated on API traffic)
|   `-- monitoring_report.json        # Output of the drift detector
|-- .github/
|   `-- workflows/
|       |-- ci.yml             # CI: lint, train model, validate artifacts, pytest, docker build
|       |-- cd.yml             # CD: Workload Identity auth, push to Artifact Registry, Cloud Run
|       `-- monitoring.yml     # Scheduled/Manual drift detection workflow
|-- Dockerfile                 # Multi-stage lightweight OCI container specification
|-- requirements.txt           # Pinned dependencies
|-- .gitignore                 # Excludes caches, venvs, and ephemeral logs
`-- README.md                  # Comprehensive classroom guide and documentation
```

---

## 3. Prerequisites & Local Installation

### Prerequisites
- **Python 3.10+** (Python 3.11 recommended)
- **Git**
- **Docker Desktop** (optional, for containerization steps)

### Setup Instructions
Open your terminal (PowerShell or Bash) and clone / navigate to the repository directory:

```bash
# Navigate to the project root
cd mlops_class

# Create a clean virtual environment
python -m venv .venv

# Activate the virtual environment:
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Step-by-Step Hands-On Walkthrough

Follow these steps in exact sequence for the classroom demonstration:

### Step 1: Train the Model & Export Artifacts
Run the deterministic training pipeline:

```bash
python -m src.train
```

**What happens behind the scenes:**
1. Generates 1,500 reproducible records using fixed seed `42` (`area_sqft`, `bedrooms`, `age_years`).
2. Calculates property price: $\text{Price} = 50,000 + (180 \times \text{area}) + (20,000 \times \text{beds}) - (1,200 \times \text{age}) + \mathcal{N}(0, 15000)$.
3. Splits into 80% training ($1,200$ rows) and 20% testing ($300$ rows).
4. Fits a Scikit-Learn `Pipeline(StandardScaler(), RandomForestRegressor())`.
5. Evaluates regression metrics on out-of-sample test data:
   - **MAE (Mean Absolute Error)**: average absolute error in dollars (~$14,100).
   - **RMSE (Root Mean Squared Error)**: penalises large outliers (~$18,000).
6. **Enforces Quality Gate**: Asserts $\text{RMSE} \le \$35,000$. If the model fails, training terminates with exit code `1`.
7. Exports artifacts into the `artifacts/` folder.

---

### Step 2: Inspect Generated Artifacts
Check the contents of the `artifacts/` folder:

```bash
# On Windows PowerShell:
Get-ChildItem artifacts/

# On Linux/macOS:
ls -lh artifacts/
```

- `house_price_pipeline.joblib`: Serialized Scikit-learn Pipeline (preprocessing + tree ensemble).
- `reference_data.csv`: Feature distributions of the training set used as baseline for drift detection.
- `metrics.json`: JSON record of training metrics and thresholds:
  ```json
  {
    "mae": 14144.33,
    "rmse": 18061.27,
    "threshold_rmse": 35000.0,
    "n_train": 1200,
    "n_test": 300,
    "random_seed": 42,
    "model_type": "Pipeline(StandardScaler, RandomForestRegressor)"
  }
  ```

---

### Step 3: Run the FastAPI Microservice
Start the local ASGI web server using Uvicorn:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Console output confirms model loading via the modern `lifespan` event:
```
[INFO] Initializing FastAPI inference service...
[INFO] Loading Scikit-learn pipeline from .../artifacts/house_price_pipeline.joblib...
[INFO] Model pipeline loaded successfully into memory.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

### Step 4: Test Endpoints via Swagger UI & cURL

#### A. Interactive OpenAPI Documentation (Swagger UI)
Open your web browser and navigate to:
```
http://127.0.0.1:8000/docs
```
You can interactively inspect schemas, test `/health`, and send test payloads to `/predict`.

#### B. Health Check Endpoint
```bash
curl -X GET "http://127.0.0.1:8000/health"
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "artifacts_path": "C:\\Users\\...\\artifacts\\house_price_pipeline.joblib"
}
```

#### C. Prediction Endpoint
Send a sample house profile:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "area_sqft": 2250.0,
       "bedrooms": 3,
       "age_years": 8.5
     }'
```
**Response:**
```json
{
  "predicted_price": 504620.15,
  "currency": "USD",
  "features": {
    "area_sqft": 2250.0,
    "bedrooms": 3,
    "age_years": 8.5
  },
  "timestamp": "2026-11-05T12:00:00.000000+00:00"
}
```

#### D. Input Validation (Safety Gate)
Try sending an invalid payload (e.g., negative area or zero bedrooms):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "area_sqft": -500.0,
       "bedrooms": 0,
       "age_years": 10.0
     }'
```
**Response (HTTP 422 Unprocessable Entity):**
FastAPI automatically rejects malformed requests before they reach the model pipeline!

#### E. View Request Logs
Every valid prediction appends a record to `artifacts/prediction_logs.csv`:
```csv
timestamp,area_sqft,bedrooms,age_years,predicted_price
2026-11-05T12:00:00.000000+00:00,2250.0,3,8.5,504620.15
```

> **Classroom Teaching Note on CSV Logging:**  
> Storing prediction logs in a local CSV is designed strictly for classroom simplicity. Production systems route telemetry asynchronously to managed streaming or cloud storage (e.g. AWS Kinesis &rarr; S3, Google Cloud Pub/Sub &rarr; BigQuery, or Datadog/Evidently AI) to prevent concurrency locks, data loss on container restart, and disk exhaustion.

---

### Step 5: Run the Automated Pytest Suite
Run the unit and integration tests:

```bash
pytest -v
```

**Expected output:**
```
tests/test_api.py::test_health_endpoint PASSED                           [ 12%]
tests/test_api.py::test_predict_endpoint_valid_payload PASSED            [ 25%]
tests/test_api.py::test_predict_validation_errors PASSED                 [ 37%]
tests/test_api.py::test_prediction_telemetry_logging PASSED              [ 50%]
tests/test_model.py::test_model_artifact_exists_and_loads PASSED         [ 62%]
tests/test_model.py::test_model_prediction_single_sample PASSED          [ 75%]
tests/test_model.py::test_model_directional_sanity PASSED                [ 87%]
tests/test_model.py::test_evaluation_metrics_artifact PASSED             [100%]
============================== 8 passed in ~8s ===============================
```

---

### Step 6: Run Data Drift Monitoring

#### Baseline (Normal Traffic)
Run the monitoring script against current live prediction logs:
```bash
python -m src.monitor
```
**Output:**
```
[INFO] Loaded reference training data with 1200 rows.
[INFO] Monitoring Result -> Overall Drift Detected: False
 - Feature 'area_sqft': Ref Mean=1988.85 | Live Mean=2120.43 | Diff=6.6% -> STABLE
 - Feature 'bedrooms':  Ref Mean=2.99    | Live Mean=2.82    | Diff=5.7% -> STABLE
 - Feature 'age_years': Ref Mean=24.90   | Live Mean=22.88   | Diff=8.1% -> STABLE
```

#### Simulating Data Drift
Simulate a scenario where incoming customer traffic shifts towards luxury mega-mansions ($3,800+$ sq ft, $5+$ bedrooms, newer age):
```bash
python -m src.monitor --simulate-drift
```
**Output:**
```
[INFO] Generating 60 simulated inference logs (drifted=True)...
[INFO] Monitoring Result -> Overall Drift Detected: True
 - Feature 'area_sqft': Ref Mean=1988.85 | Live Mean=3857.74 | Diff=94.0% -> ALERT_DRIFT_DETECTED
 - Feature 'bedrooms':  Ref Mean=2.99    | Live Mean=5.48    | Diff=83.3% -> ALERT_DRIFT_DETECTED
 - Feature 'age_years': Ref Mean=24.90   | Live Mean=5.19    | Diff=79.2% -> ALERT_DRIFT_DETECTED
```
Inspect `artifacts/monitoring_report.json` to review the structured audit log.

---

### Step 7: Build & Run with Docker

#### A. Build the Container Image
```bash
docker build -t house-price-api:latest .
```
*(Notice: The Dockerfile includes an automated build gate: `RUN test -f artifacts/house_price_pipeline.joblib || python -m src.train` which guarantees that the model is trained even if building in a fresh environment).*

#### B. Run the Container
```bash
docker run -d --name house-price-service -p 8000:8000 house-price-api:latest
```

#### C. Verify Container Health
```bash
curl http://localhost:8000/health
```

#### D. Stop and Clean Up Container
```bash
docker stop house-price-service
docker rm house-price-service
```

---

## 5. CI/CD Workflows (GitHub Actions)

### Continuous Integration (CI)
File: `.github/workflows/ci.yml`  
Triggers on: Every push and pull request to `main`.

```
[ Developer Push / PR ]
         |
         v
+-----------------------------+
| 1. Checkout & Setup Python  |
+--------------+--------------+
               |
               v
+-----------------------------+
| 2. Install Dependencies     |
+--------------+--------------+
               |
               v
+-----------------------------+
| 3. Train Model in Runner    |
| (Fails if RMSE > $35,000)   |
+--------------+--------------+
               |
               v
+-----------------------------+
| 4. Validate Model Artifacts |
| (.joblib, .csv, metrics)    |
+--------------+--------------+
               |
               v
+-----------------------------+
| 5. Run Pytest Suite         |
| (API & Model tests pass)    |
+--------------+--------------+
               |
               v
+-----------------------------+
| 6. Build Docker Image       |
| (Verifies container build)  |
+--------------+--------------+
               |
               +---> Success: Ready for CD Release
```

---

### Continuous Deployment (CD) to Google Cloud Run
File: `.github/workflows/cd.yml`  
Triggers on: Completion of CI on `main` with conclusion `success`.

#### Authentication Pattern: Workload Identity Federation (WIF)
Rather than creating long-lived, highly sensitive service account JSON keys (which risk leaks or theft), this workflow uses **Keyless OIDC Authentication** via Google Cloud Workload Identity Federation. GitHub Actions assumes a Google Cloud IAM Role ephemerally for the duration of the deployment job.

#### One-Time Google Cloud Setup Commands:
If you want to deploy to your GCP project, run these commands in the Google Cloud Shell:

```bash
# 1. Set environment variables
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export REPO_NAME="mlops-repo"
export SA_NAME="github-actions-deployer"
export POOL_NAME="github-actions-pool"
export PROVIDER_NAME="github-actions-provider"
export GITHUB_REPO="your-github-username/your-repo-name"

# 2. Enable required GCP APIs
gcloud services enable \
    artifactregistry.googleapis.com \
    run.googleapis.com \
    iam.googleapis.com \
    iamcredentials.googleapis.com

# 3. Create Artifact Registry repository
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for MLOps services"

# 4. Create Service Account for deployment
gcloud iam service-accounts create $SA_NAME \
    --display-name="GitHub Actions Deployer"

# 5. Grant necessary roles to Service Account
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"

# 6. Create Workload Identity Pool and Provider
gcloud iam workload-identity-pools create $POOL_NAME \
    --location="global" \
    --display-name="GitHub Actions Pool"

gcloud iam workload-identity-pools providers create-oidc $PROVIDER_NAME \
    --location="global" \
    --workload-identity-pool=$POOL_NAME \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository=='$GITHUB_REPO'"

# 7. Allow GitHub repository to impersonate Service Account
export POOL_ID=$(gcloud iam workload-identity-pools describe $POOL_NAME --location="global" --format="value(name)")

gcloud iam service-accounts add-iam-policy-binding "$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/iam.workloadIdentityUser" \
    --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_REPO}"

# 8. Retrieve Workload Identity Provider Resource Name
gcloud iam workload-identity-pools providers describe $PROVIDER_NAME \
    --location="global" \
    --workload-identity-pool=$POOL_NAME \
    --format="value(name)"
```

#### GitHub Repository Secrets Configuration:
Add these 4 secrets in **GitHub &rarr; Settings &rarr; Secrets and variables &rarr; Actions**:
- `GCP_PROJECT_ID`: Your GCP project ID.
- `GCP_REGION`: Target region (e.g., `us-central1`).
- `GCP_WORKLOAD_IDENTITY_PROVIDER`: Full provider resource name from step 8 above.
- `GCP_SERVICE_ACCOUNT`: `$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com`.

> **Skipping Cloud Deployment for Local Testing:**  
> If these secrets are not configured, the CD workflow safely skips deployment with an educational log message without failing your build!

---

### Scheduled Drift Monitoring Workflow
File: `.github/workflows/monitoring.yml`  
- Runs automatically on a daily schedule (`cron: "0 6 * * *"`).
- Can be triggered manually via `workflow_dispatch` (with an optional checkbox to simulate feature drift).
- Uploads `artifacts/monitoring_report.json` as a downloadable GitHub Actions artifact for compliance auditing.

---

## 6. Core MLOps Concepts Explained

### CI vs. CD vs. Continuous Training (CT)

| Phase | Acronym | Trigger | Core Responsibility | Quality Gate |
| :--- | :--- | :--- | :--- | :--- |
| **Continuous Integration** | **CI** | Git push / PR | Tests code, lints, executes pipeline training, tests API endpoints. | `pytest` passes, $\text{RMSE} \le \text{Threshold}$, Docker build succeeds. |
| **Continuous Delivery / Deployment** | **CD** | Merge to `main` (after CI passes) | Packages OCI container, pushes to registry, deploys service to Cloud Run / Kubernetes. | Health check probe (`/health` returns 200). |
| **Continuous Training** | **CT** | Scheduled cron or Data Drift alert | Retrains model on newly labelled production datasets; validates candidate vs champion. | New model must outperform existing model on fresh validation split. |

---

### The Three Layers of MLOps Monitoring

```
                    +------------------------------------------+
                    |           MLOps Telemetry Layers         |
                    +--------------------+---------------------+
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
+-----------------+             +-----------------+             +-----------------+
| 1. API & Infra  |             |  2. Data Drift  |             |3. Model Quality |
|   Monitoring    |             |   Monitoring    |             |   Monitoring    |
+-----------------+             +-----------------+             +-----------------+
| - Uptime        |             | - P(X_live) vs  |             | - MAE / RMSE    |
| - Latency (p99) |             |   P(X_baseline) |             | - Accuracy / F1 |
| - QPS           |             | - Mean shift    |             | - Error spikes  |
| - 5xx Errors    |             | - PSI / KS-test |             | - Requires True |
| - CPU / Memory  |             | - Early warning |             |   Ground Truth! |
+-----------------+             +-----------------+             +-----------------+
```

1. **Infrastructure & API Monitoring**: Measures system availability and response speeds.
2. **Data Drift Monitoring**: Compares the distribution of incoming features against the training baseline ($P(X_{\text{live}})$ vs $P(X_{\text{ref}})$). Warns engineers of changes in user demographics or inputs *before ground truth is known*.
3. **Model Quality Monitoring**: Computes true prediction error ($\text{RMSE}$, $\text{MAE}$, $\text{R}^2$). Evaluates whether the model is still making accurate predictions.

---

### The Ground-Truth Feedback Delay
In classification on web clicks or recommendations, ground truth is known in seconds.  
However, in **real estate house prices**, a house predicted today might take **90 days** to sell.  
Therefore:
- You cannot compute live MAE or RMSE on day 1 because the true sale price does not exist yet!
- **Data drift monitoring is your early warning defense**: If users start querying $6,000$ sq ft houses when your training data only went up to $4,500$ sq ft, drift alerts notify you of out-of-distribution extrapolations before customers complain about bad predictions.

---

### Why Pipelines & Single Artifacts Matter
In amateur ML projects, feature scaling (e.g., `StandardScaler`) is computed in a notebook, while raw features are sent to the API. This creates catastrophic **data leakage** and **training-serving skew**.

In this project, we package preprocessing and estimator into a single Scikit-Learn `Pipeline`:
```python
Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(...))
])
```
When serialized into `house_price_pipeline.joblib`:
- Preprocessing mean and standard deviations are saved alongside model weights.
- The FastAPI `/predict` endpoint feeds raw, unscaled JSON features directly to `pipeline.predict()`.
- Preprocessing is identical in training, testing, and production.

---

### Model Governance & Safe Promotion

> [!IMPORTANT]
> **A retrained model should NEVER automatically replace the production model.**  
> Continuous Training (CT) produces a *candidate model*. Before promotion:
> 1. Candidate must pass automated regression quality gates ($\text{RMSE} \le \text{Threshold}$).
> 2. Candidate must outperform the existing "Champion" model on recent hold-out benchmark datasets.
> 3. Candidate must undergo Canary or Shadow Deployment (traffic split) to observe live latency and sanity.
> 4. Formal approval / sign-off must be recorded before updating the production endpoint.

---

## Classroom Demonstration Checklist

When presenting this project to students, follow this 10-step sequence:
1. Walk through the folder structure and explain the purpose of each file.
2. Review `src/train.py`: Show how synthetic features are generated and explain the Scikit-learn Pipeline.
3. Run `python -m src.train` and inspect `artifacts/metrics.json`.
4. Launch the API with `uvicorn app.main:app --reload`.
5. Open `http://127.0.0.1:8000/docs` in the browser; execute `/health` and `/predict`.
6. Demonstrate input validation by sending invalid data (negative area) to see the 422 HTTP response.
7. Run the test suite with `pytest -v` and explain how unit and API tests prevent regressions.
8. Run `python -m src.monitor` to show baseline statistics, then `python -m src.monitor --simulate-drift` to demonstrate drift detection.
9. Review `.github/workflows/ci.yml` and `.github/workflows/cd.yml` to explain CI checks, WIF security, and deployment gates.
10. Conclude with the summary: **CI tests code & models &rarr; CD deploys services &rarr; Monitoring tracks telemetry &rarr; CT retrains when data shifts**.

