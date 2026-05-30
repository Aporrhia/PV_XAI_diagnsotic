"""
Core Engine Utilities for PV XAI Diagnostics.

Provides XAI model tournament selection, global SHAP values,
hybrid ARIMA baseline forecasting, and automated LLM reporting.
"""

# pylint: disable=import-error
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from google import genai
from google.genai import types
from lightgbm import LGBMRegressor
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

from preprocessing import PVPreprocessor

# Global client instantiation
client = genai.Client(api_key="NO_KEY")


# pylint: disable=too-many-locals, invalid-name
def run_autonomous_selection(df, features, target, models_to_run):
    """
    Train and evaluate a set of ML algorithms, returning a leaderboard and the best config.
    """
    X = df[features]
    y = df[target].clip(lower=0)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)

    model_pool = {
        "XGBoost": XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=8, objective='reg:absoluteerror'
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, criterion='absolute_error', max_depth=15
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            loss='poisson', max_iter=800, max_leaf_nodes=63
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=800, objective='huber', learning_rate=0.05
        ),
        "CatBoost": CatBoostRegressor(
            iterations=800, loss_function='MAE', depth=8, verbose=0
        )
    }

    results = []
    best_model = None
    best_r2 = -np.inf

    for name in models_to_run:
        if name in model_pool:
            model = model_pool[name]
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)

            results.append({"Model": name, "R2": round(r2, 4), "MAE": round(mae, 4)})

            if r2 > best_r2:
                best_r2 = r2
                best_model = {
                    "name": name,
                    "model": model,
                    "r2": r2,
                    "X_test": X_test_scaled,
                    "scaler": scaler
                }

    return pd.DataFrame(results), best_model


# pylint: disable=invalid-name
def get_shap_explainer(model, X):
    """
    Generates SHAP values using a balanced sample to prevent computational hangs.
    """
    # Limit to 300 rows for the global plot
    X_sample = shap.sample(X, min(len(X), 300))
    explainer = shap.TreeExplainer(model)

    # Calculate values ONLY for the sample
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values, X_sample


def generate_diagnostic_report(prediction, method_name, contributions, anomaly_context, sensor_data):
    """
    Submit raw SCADA contexts and XAI feature attributions to Gemini to draft engineering report.
    """
    impact_summary = "\n".join([f"- {feat}: {impact:.4f} impact" for feat, impact in contributions])

    # Format the Raw Sensor Data without NaNs
    sensor_strings = []
    for col, val in sensor_data.items():
        if pd.notna(val):
            sensor_strings.append(f"{col}: {val}")
    sensor_context = " | ".join(sensor_strings)

    # System Prompt
    system_instruction = (
        "You are a professional Solar PV Diagnostics Engineer. Your objective is to translate "
        "complex SCADA and XAI data into a mathematically transparent root-cause analysis.\n\n"
        "SITE-SPECIFIC ELECTRICAL CONTEXT:\n"
        "- Current THD (thdI_GEN) is load-dependent. It is naturally HIGHEST (often 15% to 50%) "
        "when the system is operating at less than 10% output.\n"
        "- Do NOT diagnose a hardware fault based solely on thdI_GEN if the system is in a "
        "low-generation state (early morning/late evening).\n"
        "- Hardware Profiles: Maple Drive East and Forest Road use SMA Sunny Boy inverters; "
        "YMCA uses Mastervolt Soladin.\n\n"
        "DIAGNOSTIC HIERARCHY & BENCHMARKS:\n"
        "1. Atmospheric Transients: If 'SolarRad' is significantly lower than 'HiSolarRad' "
        "(>20% gap), identify cloud transients as the primary factor reducing potential.\n"
        "2. Grid Throttling: If 'VA_Rise', 'VB_Rise', or 'VC_Rise' exceed 2.0V, the inverter is "
        "likely throttling to manage local grid voltage.\n"
        "3. Voltage Quality (IEEE 519): If 'thdV_Filtered' exceeds 5%, flag a confirmed power "
        "quality violation regardless of load.\n"
        "4. Current Quality (IEEE 1547): Treat 5% as the 'rated output' target. For diagnostic "
        "purposes in this dataset, only suspect a fault if thdI_GEN exceeds 50% or if a spike "
        ">15% occurs during peak generation (>50% capacity).\n"
        "5. Chronic Maintenance Issues: If SolarRad is near HiSolarRad and electrical metrics "
        "are healthy, but P_GEN is >30% below the ML Physics baseline, diagnose as string failure, "
        "severe soiling, or localized shading.\n\n"
        "REPORT FORMAT:\n"
        "- Paragraph 1: Quantitative Analysis. Juxtapose the Theoretical ML Prediction, the "
        "Historical ARIMA Routine, and the Actual Sensor Output. Use these to define the "
        "'Generation Gap.'\n"
        "- Paragraph 2: Root Cause Diagnosis. Provide a weighted explanation using the benchmarks "
        "above. Clearly state if the issue is Atmospheric (weather) or Mechanical "
        "(hardware/maintenance).\n"
        "- STRICTURE: Do not use markdown bolding (**) in the output text. Keep the tone professional "
        "and authoritative."
    )

    user_message = f"""
    Anomaly Context: {anomaly_context}

    Raw Sensor Data at this timestamp:
    {sensor_context}

    The AI model predicted {prediction:.2f} kW.
    The {method_name} XAI explanation highlights these primary drivers pushing the prediction:
    {impact_summary}

    Based on the expert ruleset and this data, what is the specific diagnosis?
    """

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            )
        )
        return response.text

    except Exception:  # pylint: disable=broad-exception-caught
        return "Diagnostic generation failed due to API error."


def get_gemini_response(messages, system_prompt, temperature=0.7):
    """
    General purpose conversation utility wrapper for chat execution layout.
    """
    # Base prompt if no system prompt provided as backup
    base_prompt = "You are a helpful solar energy assistant."

    try:
        # Prepare the history
        formatted_contents = []
        for msg in messages[:-1]:
            role = "model" if msg["role"] == "assistant" else "user"
            formatted_contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        # Append the LATEST message
        formatted_contents.append({
            "role": "user",
            "parts": [{"text": messages[-1]["content"]}]
        })

        system_instruction = system_prompt[0]["content"] if system_prompt else base_prompt
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=formatted_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
            )
        )

        return {"message": {"content": response.text}}

    except Exception:  # pylint: disable=broad-exception-caught
        return {"message": {"content": "Gemini Error"}}


def run_hybrid_arima(historical_series):
    """
    Evaluate historic load structures using a 1-step ahead ARIMA prediction model.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore convergence warnings for speed
        try:
            model = ARIMA(historical_series, order=(2, 1, 2))
            fitted = model.fit()
            # Forecast 1 step ahead
            forecast = fitted.forecast(steps=1)
            return forecast.iloc[0]
        except Exception:  # pylint: disable=broad-exception-caught
            return None


# pylint: disable=too-few-public-methods
class ModelSelector:
    """Lightweight wrapper providing a class interface for autonomous model selection.

    Attributes:
        model_pool: mapping of model name to estimator instance
    """
    model_pool: Dict[str, Any]
    preprocessor: Optional[PVPreprocessor] = None

    def __init__(self) -> None:
        self.model_pool = {}
        self.preprocessor = None

    def select(self, df: pd.DataFrame, features: List[str], target: str, models_to_run: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Models execution pipeline loop using dataset arrays."""
        return run_autonomous_selection(df, features, target, models_to_run)


# pylint: disable=too-few-public-methods
class HybridForecaster:
    """Wrapper for ARIMA forecasting used in hybrid diagnostics."""
    def __init__(self) -> None:
        pass

    def forecast(self, historical_series: pd.Series) -> Optional[float]:
        """Execute linear historical regression mapping on series indices."""
        return run_hybrid_arima(historical_series)


# pylint: disable=too-few-public-methods, too-many-arguments, too-many-positional-arguments
class DiagnosticGenerator:
    """Wrapper around LLM diagnostic report generation.

    Attributes:
        client: client used to call LLM APIs
    """
    client: genai.Client

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.client = client
        if api_key:
            # set client key if provided
            self.client.api_key = api_key

    def generate(self, prediction: float, method_name: str, contributions: List[tuple], anomaly_context: str, sensor_data: Dict[str, Any]) -> str:
        """Call global translation API layout string formatting."""
        return generate_diagnostic_report(prediction, method_name, contributions, anomaly_context, sensor_data)