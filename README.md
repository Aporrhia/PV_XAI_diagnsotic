# PV XAI Diagnostic Portal

## Introduction

PV XAI Diagnostic Portal is a Streamlit application for photovoltaic (PV) fault detection and diagnostics. It ingests PV production and weather data, auto-selects physics-aware features, runs a model tournament, and provides explainable AI (SHAP/LIME) insights with a hybrid ML + ARIMA diagnostic workflow. Finally, the report is generated, if the Gemini API key is provided.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) If you plan to use the LLM diagnostic report generation, set a Gemini API key in code or update the client initialisation in [engine.py](engine.py).

## Usage

### Run the Streamlit app

```bash
streamlit run app.py
```

### In the UI

1. Upload PV and weather data files (CSV or Excel).
2. Choose the target variable (default: `P_GEN`).
3. (Engineers) Select features and models or use auto-selection.
4. Run the pipeline to train, validate, and review XAI-based global explanations.
5. Run the hybrid diagnostic to compare ML predictions with ARIMA baselines for the selected anomaly.
6. (Optional) If the Gemini API key was provided, then the report will be generated based on the anomaly details.

### Example data

A sample dataset is available at [Datasets/pv_data_10min.csv](Datasets/pv_data_10min.csv) and [Datasets/weather_data.xlsx](Datasets/weather_data.xlsx).
