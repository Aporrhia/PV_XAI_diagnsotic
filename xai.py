import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt
import warnings

def run_xai_comparison(model, X_train, X_test, y_test=None):
    # 1. Drop features with near-zero variance
    variances = X_train.var()
    low_variance_features = variances[variances < 1e-6].index.tolist()
    
    if low_variance_features:
        print(f"Dropping near-constant features for XAI stability: {low_variance_features}")
        # FIX: Add errors='ignore' so it doesn't crash if X_test is already cleaned
        X_train = X_train.drop(columns=low_variance_features, errors='ignore')
        X_test = X_test.drop(columns=low_variance_features, errors='ignore')

    # Align features between the model and the provided datasets to avoid
    # "feature names should match" errors when calling predict/explainers.
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
    else:
        model_features = list(X_train.columns)

    # Reindex both training and test sets to the model's feature order.
    # Missing columns are filled with zeros (safe fallback for explanations),
    # extra columns are dropped.
    X_train = X_train.reindex(columns=model_features, fill_value=0)
    X_test = X_test.reindex(columns=model_features, fill_value=0)

    # 2. SHAP (Global Explanation)
    print("Running SHAP Explainer...")
    X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
    explainer_shap = shap.Explainer(model.predict, X_sample)
    shap_values = explainer_shap(X_sample)

    plt.figure(figsize=(10, 6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("Global Physics Impact (Atmospheric Drivers)")
    plt.tight_layout()
    plt.savefig("xai_shap_global.png")
    plt.close()

    # 3. LIME (Local Explanation of a 'Drop')
    print("Running LIME Explainer...")
    try:
        # TARGETED SEARCH: Find a row where energy dropped despite sunshine
        if y_test is not None and 'SolarRad' in X_test.columns:
            # Efficiency proxy: Solar is high, Power is low
            efficiency = X_test['SolarRad'] / (y_test + 0.1)
            problem_indices = efficiency.sort_values(ascending=False).head(20).index
            idx_name = problem_indices[0]
            idx = X_test.index.get_loc(idx_name)
            print(f"Explaining drop at {idx_name} (High Solar, Low Output)")
        else:
            idx = 0

        # Custom predict to silence 'feature names' warning
        def custom_predict(data_array):
            df_temp = pd.DataFrame(data_array, columns=X_train.columns)
            return model.predict(df_temp)

        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            mode='regression',
            # Set this to True but use 'quartile' to avoid the scale/truncnorm error
            discretize_continuous=True, 
            discretizer='quartile', 
            random_state=42
        )
        
        data_row = X_test.iloc[idx].values 
        exp = explainer_lime.explain_instance(
            data_row, 
            custom_predict,
            num_features=10,
            num_samples=5000
        )
        
        fig = exp.as_pyplot_figure()
        plt.title(f"Why did power drop at {X_test.index[idx]}?")
        plt.tight_layout()
        plt.savefig("xai_lime_local.png")
        plt.close()
        print("XAI Comparison Complete. Check PNG files.")
        
    except Exception as e:
        print(f"LIME failed: {e}")