# AeroSentinel

**AI/ML-Based Drone Telemetry Analytics, Anomaly Detection & Remaining Flight-Time Prediction**

AeroSentinel is a machine-learning project for analyzing multivariate drone telemetry, identifying abnormal flight behavior, and estimating remaining flight time. It currently supports **Holybro** and **SpeedyBee** hardware configurations.

## Project Overview

AeroSentinel brings together telemetry processing, machine-learning inference, and an interactive analytics interface.

### Implemented capabilities

- Telemetry data processing and feature engineering
- Exploratory data analysis and processed telemetry datasets
- Isolation Forest anomaly detection for Holybro and SpeedyBee
- Holybro remaining-flight-time prediction
- FastAPI inference service
- Streamlit analytics dashboard
- Prediction logging and monitoring endpoints
- Automated Python tests and GitHub Actions CI

### Machine-learning models

The trained model files are stored in `models/`.

| Model file | Purpose |
|---|---|
| `holybro_isolation_forest.joblib` | Holybro anomaly detection |
| `speedybee_isolation_forest.joblib` | SpeedyBee anomaly detection |
| `holybro_remaining_time_model.joblib` | Holybro remaining-time prediction |
| `holybro_future_remaining_time_30s.joblib` | Holybro future remaining-time prediction |

The current remaining-time forecasting models are for Holybro. SpeedyBee anomaly detection is supported, but SpeedyBee remaining-time forecasting is not currently supported by the API.

## Dataset

The complete project dataset is hosted on Hugging Face:

**[AeroSentinel Dataset — Hugging Face](https://huggingface.co/datasets/Sagarsing007/AeroSentinel)**

Dataset information:

- Approximately 986 MB uploaded
- 269 uploaded files
- Holybro and SpeedyBee telemetry
- Processed telemetry and feature datasets
- Metadata, dictionaries, and analysis outputs

The dataset is hosted separately from the GitHub source repository. The local `data/` directory is excluded from Git.

## Repository Structure

```text
AeroSentinel/
├── .github/
│   └── workflows/
│       └── ci.yml
├── dashboard/
│   ├── app.py
│   └── testapp.py
├── models/
│   ├── holybro_future_remaining_time_30s.joblib
│   ├── holybro_isolation_forest.joblib
│   ├── holybro_remaining_time_model.joblib
│   └── speedybee_isolation_forest.joblib
├── reports/
├── src/
│   ├── api.py
│   ├── inference_engine.py
│   ├── prediction_logger.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_inference.py
│   ├── evaluate_remaining_time_fallback.py
│   └── validate_holybro_normalization.py
├── data/                 # Downloaded locally; excluded from Git
├── requirements.txt
├── run_pipeline.py
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SagarKarosiya/AeroSentinel.git
cd AeroSentinel
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install project dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install additional dashboard and data-processing dependencies

The dashboard and optional Parquet processing require packages not currently listed in the base requirements file.

```bash
pip install streamlit requests plotly joblib pyarrow
```

For running the automated tests:

```bash
pip install pytest httpx
```

## Dataset Setup

The dataset is hosted on Hugging Face and should be downloaded into the repository root so its `data/` directory is restored.

Install the Hugging Face CLI:

```bash
pip install -U huggingface_hub
```

Download the dataset:

```bash
hf download Sagarsing007/AeroSentinel --repo-type dataset --local-dir .
```

After downloading, check that the expected local data directories exist, including:

```text
data/
├── processed/
└── features/
```

The exact files available in these directories depend on the published dataset revision.

## Running the Application

### 1. Start the FastAPI inference service

From the repository root, with the virtual environment activated:

```bash
python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

API documentation:

- Interactive Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- Health check: http://127.0.0.1:8000/health

The API currently exposes anomaly detection, remaining-time prediction, and monitoring endpoints.

Supported hardware values:

- `holybro`
- `speedybee`

### 2. Start the Streamlit dashboard

Open a second terminal, activate the same virtual environment, and run:

```bash
streamlit run dashboard/app.py
```

The dashboard is configured to communicate with the API at:

```text
http://127.0.0.1:8000
```

Start the API before using dashboard features that require inference.

## Running the Data Pipeline

The repository includes a pipeline script:

```bash
python run_pipeline.py
```

The current script:

1. Scans CSV files under `data/processed`.
2. Identifies telemetry files containing a `timestamp` column.
3. Aligns and resamples telemetry to a target frequency of 10 Hz.
4. Saves aligned data as Parquet files under `data/aligned`.
5. Runs the LSTM autoencoder training script if telemetry files were successfully processed.

This pipeline requires the local dataset and a compatible Parquet engine. Review the script and its input assumptions before running it on a new dataset.

## Running Automated Tests

Run the test suite from the repository root:

```bash
python -m pytest -v src
```

The automated tests cover:

- API health endpoint
- Holybro anomaly inference
- SpeedyBee anomaly inference
- Holybro remaining-time inference
- Unsupported SpeedyBee remaining-time request handling

The dataset validation and remaining-time fallback evaluation scripts are separate manual scripts. They are not part of the pytest collection.

## Continuous Integration

A GitHub Actions workflow is configured at:

```text
.github/workflows/ci.yml
```

It performs Python syntax validation, automated tests, and essential-file checks on pushes and pull requests targeting `main`.

**[View AeroSentinel GitHub Actions](https://github.com/SagarKarosiya/AeroSentinel/actions)**

## Technology Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn, PyTorch
- **Model persistence:** Joblib
- **API:** FastAPI, Uvicorn
- **Dashboard:** Streamlit, Plotly
- **Telemetry ingestion:** PyULog
- **Data storage:** CSV, Parquet
- **Experiment tracking / data versioning dependencies:** MLflow, DVC
- **CI:** GitHub Actions
- **Dataset hosting:** Hugging Face

## Current Project Status

The repository currently includes trained anomaly-detection models for both supported hardware configurations, Holybro remaining-time models, a FastAPI inference service, a Streamlit dashboard, and an automated CI test workflow.

Further development areas include broader cross-hardware forecasting, expanded model architectures and evaluation, explainability, edge optimization, and production deployment.

These areas should be considered ongoing development until their implementations and results are independently verified.

## Project Links

- **GitHub Repository:** https://github.com/SagarKarosiya/AeroSentinel
- **Hugging Face Dataset:** https://huggingface.co/datasets/Sagarsing007/AeroSentinel
- **GitHub Actions:** https://github.com/SagarKarosiya/AeroSentinel/actions

---

**AeroSentinel — Drone Telemetry Intelligence & Predictive Analytics.**
