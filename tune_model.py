"""
Hyperparameter Optimisation and Model Tuning.
"""

from typing import Any, Dict, Optional, Tuple
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from engine import ModelSelector


def prepare_for_training(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply engineering transformations to raw dataset tables.

    Args:
        df: Input merged weather and production metrics dataframe.

    Returns:
        Tuple containing a feature DataFrame and target series.
    """
    df = df[df['SolarRad'] > 5].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = 'timestamp' if 'timestamp' in df.columns else 'datetime'
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)

    hours = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24)

    # Peak Sun Hour Flag
    df['is_peak_sun'] = ((hours >= 10) & (hours <= 14)).astype(int)

    # Solar Gradient
    df['Solar_Gradient'] = df['SolarRad'].diff().fillna(0)

    # Moving Averages & Interactions
    df['Solar_10min_Avg'] = df['SolarRad'].rolling(window=3).mean().fillna(df['SolarRad'])
    df['Temp_Solar_Interaction'] = df['SolarRad'] * df['TempOut']

    if 'HiSolarRad' in df.columns:
        df['Cloud_Factor'] = (df['HiSolarRad'] - df['SolarRad']).clip(lower=0)

    features = [
        'SolarRad', 'TempOut', 'WindSpeed', 'OutHum',
        'hour_sin', 'hour_cos', 'is_peak_sun', 'Solar_Gradient',
        'Solar_10min_Avg', 'Temp_Solar_Interaction', 'Cloud_Factor',
        'Substation_ID', 'VA_Rise', 'VC_Rise'
    ]

    existing_features = [f for f in features if f in df.columns]
    x_df = df[existing_features]
    y_series = df['P_GEN'].clip(lower=0)

    return x_df, y_series


def autonomous_model_selection(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame, pd.Series]:
    """
    Train and score models to determine the optimal starting estimator configuration.

    Args:
        df: Base input historical telemetry frame.

    Returns:
        Tuple containing a evaluation dictionary, test features, and test targets.
    """
    x_df, y_series = prepare_for_training(df)
    numeric_cols = x_df.select_dtypes(include=[np.number])
    x_df = numeric_cols.replace([np.inf, -np.inf], np.nan).dropna()
    y_series = y_series.loc[x_df.index].clip(lower=0)

    split_idx = int(len(x_df) * 0.8)
    x_train, x_test = x_df.iloc[:split_idx], x_df.iloc[split_idx:]
    y_train, y_test = y_series.iloc[:split_idx], y_series.iloc[split_idx:]

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

    models = {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.03, max_depth=8, objective='reg:absoluteerror'
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.03, objective='poisson', importance_type='gain'
        ),
        "CatBoost": cb.CatBoostRegressor(
            iterations=1000, learning_rate=0.03, depth=8, loss_function='MAE', verbose=0
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=1000, loss='poisson', max_depth=12
        )
    }

    best_info = {
        "model": None, "name": None, "score": -np.inf, "R2": None, "MAE": None, "scaler": scaler
    }

    for name, model in models.items():
        model.fit(x_train_scaled, y_train)
        preds = model.predict(x_test_scaled)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        # BLENDED SCORE - Favors high R2 but rewards low MAE equally
        current_score = r2 * (1 / (mae + 1))

        if current_score > best_info["score"]:
            best_info.update({
                "score": current_score, "R2": r2, "MAE": mae, "name": name, "model": model
            })

    return best_info, x_test_scaled, y_test


def optimise_winner(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Optimizes the winning model architecture using cross-validation."""
    model_name = model.__class__.__name__
    
    # 1. Create a clean validation split from the training data (Temporal Split)
    # This acts as our final gatekeeper to test the models before accepting changes.
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    # 2. Establish a baseline by training a clean clone of the base model
    try:
        base_model = clone(model)
    except Exception:
        base_model = model # Fallback if cloning is unsupported
        
    base_model.fit(X_tr, y_tr)
    preds_base = base_model.predict(X_val)
    r2_base = r2_score(y_val, preds_base)
    mae_base = mean_absolute_error(y_val, preds_base)

    # 3. Define stable parameter search spaces
    tscv = TimeSeriesSplit(n_splits=3)
    param_distributions = {}

    if "XGBRegressor" in model_name:
        param_distributions = {
            'max_depth': [3, 5, 6],
            'learning_rate': [0.03, 0.05, 0.1],
            'n_estimators': [100, 200, 400]
        }
    elif "LGBMRegressor" in model_name:
        param_distributions = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'verbose': [-1]
        }
    elif "CatBoostRegressor" in model_name:
        param_distributions = {
            'depth': [4, 6],
            'learning_rate': [0.03, 0.05],
            'l2_leaf_reg': [3, 5],
            'iterations': [100, 200],
            'verbose': [0] 
        }
    else:
        # If model doesn't match tree pool, return baseline immediately
        return model

    # 4. Execute Hyperparameter Search
    try:
        # Run on the model instance without an extra scaling pipeline step
        search = RandomizedSearchCV(
            clone(model),
            param_distributions=param_distributions,
            n_iter=4, 
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42
        )
        search.fit(X_tr, y_tr)
        tuned_candidate = search.best_estimator_
        
        # Evaluate how the tuned candidate performs on unseen validation data
        preds_tuned = tuned_candidate.predict(X_val)
        r2_tuned = r2_score(y_val, preds_tuned)
        mae_tuned = mean_absolute_error(y_val, preds_tuned)
        tol_r2 = 0.02
        
        print(f"[{model_name} Evaluation Summary]")
        print(f"  -> Base Model  - R2: {r2_base:.4f} | MAE: {mae_base:.4f}")
        print(f"  -> Tuned Model - R2: {r2_tuned:.4f} | MAE: {mae_tuned:.4f}")

        if (r2_tuned >= r2_base - tol_r2) and (r2_tuned > 0) and (mae_tuned < mae_base * 1.5):
            print(f" Verification passed. Deploying tuned parameters for {model_name}.")
            # Retrain on the complete training set before returning
            tuned_candidate.fit(X_train, y_train)
            return tuned_candidate
        else:
            print(f" Tuning degraded performance or diverged. Reverting safely to baseline.")
            model.fit(X_train, y_train)
            return model

    except Exception as e:
        print(f" Tuning execution failed for {model_name}: {e}. Preserving baseline.")
        model.fit(X_train, y_train)
        return model


class ModelTuner:
    """
    Wrapper for model tuning and selection routines.

    Attributes:
        selector: Optional internal structural tracking link mapping to model pool selections.
    """

    selector: Optional[ModelSelector] = None

    def __init__(self) -> None:
        """Instantiate empty configurations."""
        self.selector = None

    def prepare(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Expose training preprocessors."""
        return prepare_for_training(df)

    def select(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame, pd.Series]:
        """Expose selection pipelines."""
        return autonomous_model_selection(df)

    def optimise(self, model: Any, x_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Expose optimisation logic."""
        return optimise_winner(model, x_train, y_train)