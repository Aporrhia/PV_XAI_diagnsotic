"""
XAI Diagnostic Evaluation Module.

Provides comparative visualisation layers utilizing SHAP and LIME
for interpreting ML predictive behaviors on atmospheric driver data.
"""

import warnings
from typing import Any, Optional
import pandas as pd
import lime
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt

# pylint: disable=import-error
from engine import ModelSelector


# pylint: disable=too-many-locals
def run_xai_comparison(model, X_train, X_test, y_test=None):
    """
    Execute SHAP and LIME explainable AI algorithms on models.
    """
    # 1. Drop features with near-zero variance
    variances = X_train.var()
    low_variance_features = variances[variances < 1e-6].index.tolist()

    if low_variance_features:
        print(f"Dropping near-constant features for XAI stability: {low_variance_features}")
        X_train = X_train.drop(columns=low_variance_features, errors='ignore')
        X_test = X_test.drop(columns=low_variance_features, errors='ignore')

    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
    else:
        model_features = list(X_train.columns)

    X_train = X_train.reindex(columns=model_features, fill_value=0)
    X_test = X_test.reindex(columns=model_features, fill_value=0)

    # 2. SHAP (Global Explanation)
    print("Running SHAP Explainer...")
    # pylint: disable=invalid-name
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

    # 3. LIME (Local Explanation)
    print("Running LIME Explainer...")
    try:
        # Find a row where energy dropped despite sunshine
        if y_test is not None and 'SolarRad' in X_test.columns:
            # Solar is high, Power is low
            efficiency = X_test['SolarRad'] / (y_test + 0.1)
            problem_indices = efficiency.sort_values(ascending=False).head(20).index
            idx_name = problem_indices[0]
            idx = X_test.index.get_loc(idx_name)
            print(f"Explaining drop at {idx_name} (High Solar, Low Output)")
        else:
            idx = 0

        def custom_predict(data_array):
            df_temp = pd.DataFrame(data_array, columns=X_train.columns)
            return model.predict(df_temp)

        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            mode='regression',
            # Using 'quartile' to avoid the scale/truncnorm error
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

        exp.as_pyplot_figure()
        plt.title(f"Why did power drop at {X_test.index[idx]}?")
        plt.tight_layout()
        plt.savefig("xai_lime_local.png")
        plt.close()
        print("XAI Comparison Complete. Check PNG files.")

    except Exception as e:
        print(f"LIME failed: {e}")


# pylint: disable=invalid-name
class XAIComparer:
    """Class wrapper for the XAI comparison utilities."""
    model: Any
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: Optional[pd.Series]

    model_selector: Optional[ModelSelector] = None

    def __init__(self, model: Any = None, X_train: pd.DataFrame = None, X_test: pd.DataFrame = None, y_test: Optional[pd.Series] = None) -> None:
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_selector = None

    def compare(self) -> None:
        """Run the XAI comparison."""
        if self.model is None or self.X_train is None or self.X_test is None:
            raise ValueError("model, X_train and X_test must be provided")
        return run_xai_comparison(self.model, self.X_train, self.X_test, self.y_test)