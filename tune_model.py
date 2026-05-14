import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def prepare_for_training(df):
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
    X = df[existing_features]
    y = df['P_GEN'].clip(lower=0)
    
    return X, y

def autonomous_model_selection(df):
    X, y = prepare_for_training(df)
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index].clip(lower=0)
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    models = {
        "XGBoost": xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=8, objective='reg:absoluteerror'),
        "LightGBM": LGBMRegressor(n_estimators=1000, learning_rate=0.03, objective='poisson', importance_type='gain'),
        "CatBoost": CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=8, loss_function='MAE', verbose=0),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=1000, loss='poisson', max_depth=12)
    }
    
    best_info = {"model": None, "name": None, "score": -np.inf, "R2": None, "MAE": None, "scaler": scaler}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        # BLENDED SCORE - Favors high R2 but rewards low MAE equally
        current_score = r2 * (1 / (mae + 1)) 

        if current_score > best_info["score"]:
            best_info.update({"score": current_score, "R2": r2, "MAE": mae, "name": name, "model": model})

    return best_info, X_test_scaled, y_test

def optimise_winner(model, X_train, y_train):
    y_train = y_train.clip(lower=0)
    tscv = TimeSeriesSplit(n_splits=3)

    candidate_params = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 500, 1000],
        'max_iter': [500, 1000],
        'max_depth': [5, 10, 20],
        'l2_regularization': [0.0, 0.1, 1.0],
        'max_leaf_nodes': [31, 63, 127],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Keep only params supported by this estimator
    valid_keys = set(model.get_params().keys())
    param_distributions = {k: v for k, v in candidate_params.items() if k in valid_keys}

    # If nothing valid, skip hyperparameter search
    if not param_distributions:
        return model

    try:
        search = RandomizedSearchCV(
            model,
            param_distributions=param_distributions,
            n_iter=10,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=0
        )
        search.fit(X_train, y_train)
        return search.best_estimator_
    except Exception:
        return model