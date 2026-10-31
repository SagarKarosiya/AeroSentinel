# AeroSentinel

**AI/ML-Based Drone Telemetry Analytics, Anomaly Detection & Remaining Flight-Time Prediction**

AeroSentinel is a machine learning project designed to analyze drone telemetry data, detect abnormal flight behavior, and predict remaining flight time using telemetry features.

## Dataset

The complete AeroSentinel dataset is publicly hosted on Hugging Face.

**[Access the AeroSentinel Dataset](https://huggingface.co/datasets/Sagarsing007/AeroSentinel)**

Dataset details:
- 269 uploaded files
- Approximately 986 MB uploaded
- Holybro and SpeedyBee telemetry
- Processed telemetry, feature datasets, metadata and analysis outputs

## Key Features

- Drone telemetry data processing
- Exploratory Data Analysis (EDA)
- Data cleaning and feature engineering
- Anomaly detection using Isolation Forest
- Remaining flight-time prediction
- Model evaluation and visualization
- Interactive analytics dashboard

## Machine Learning Models

The trained models are available in the `models/` directory.

| Model | Purpose |
|---|---|
| `holybro_future_remaining_time_30s.joblib` | Holybro future remaining-time prediction |
| `holybro_isolation_forest.joblib` | Holybro anomaly detection |
| `holybro_remaining_time_model.joblib` | Holybro remaining-time prediction |
| `speedybee_isolation_forest.joblib` | SpeedyBee anomaly detection |

## Project Structure

```text
AeroSentinel/
├── dashboard/
├── data/
├── models/
├── reports/
├── src/
├── requirements.txt
├── run_pipeline.py
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/SagarKarosiya/AeroSentinel.git
cd AeroSentinel
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Access

Download or explore the complete dataset here:

https://huggingface.co/datasets/Sagarsing007/AeroSentinel

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Machine Learning
- Parquet
- Streamlit

## Project Links

- **GitHub Repository:** https://github.com/SagarKarosiya/AeroSentinel
- **Hugging Face Dataset:** https://huggingface.co/datasets/Sagarsing007/AeroSentinel

---

**AeroSentinel — Drone Telemetry Intelligence & Predictive Analytics.**