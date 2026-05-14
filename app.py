import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import time
import lime
import lime.lime_tabular
import warnings
import os
from engine import run_autonomous_selection, get_shap_explainer, run_hybrid_arima, generate_diagnostic_report
from preprocessing import PVPreprocessor
from tune_model import optimise_winner
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error

# --- PAGE CONFIG ---
st.set_page_config(page_title="PV-XAI Autonomous Portal", layout="wide")
st.title("Autonomous FDD & Diagnostic Portal")

# --- SESSION STATE INITIALIZATION ---
if 'selected_features' not in st.session_state:
    st.session_state['selected_features'] = []
if 'training_done' not in st.session_state:
    st.session_state['training_done'] = False

# --- ROLE AUTHENTICATION ---
st.sidebar.header("User Role")
user_role = st.sidebar.radio(
    "Select your role:", 
    ["Maintenance Engineer", "Grid Stakeholder"],
    help="Determines the level of technical metrics displayed."
)
is_engineer = (user_role == "Maintenance Engineer")

# --- HELPER FUNCTIONS ---
def auto_select_features(df, target_col):
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()[target_col].abs()
    physics_keywords = ['solar', 'rad', 'temp', 'wind', 'hum', 'bar', 'dew', 'hi']
    suggested = [col for col in numeric_df.columns if any(k in col.lower() for k in physics_keywords)]
    leaky_cols = correlations[correlations > 0.98].index.tolist()
    final_suggestions = [c for c in suggested if c not in leaky_cols and c != target_col]
    return final_suggestions, leaky_cols

@st.cache_data
def process_uploaded_files(f1, f2):
    try:
        prep = PVPreprocessor()
        df1 = pd.read_csv(f1) if f1.name.endswith('.csv') else pd.read_excel(f1, engine='openpyxl')
        df2 = pd.read_csv(f2) if f2.name.endswith('.csv') else pd.read_excel(f2, engine='openpyxl')
        
        if 'P_GEN' in df1.columns or 'datetime' in df1.columns:
            prep.pv_data, prep.weather_data = df1, df2
        else:
            prep.pv_data, prep.weather_data = df2, df1
            
        prep.clean_pv_data()
        prep.clean_weather_data()
        prep.merge_datasets(tolerance='10min')
        prep.encode_categoricals()
        prep.handle_nulls(inplace=True)
        return prep.merged_data
    except Exception as e:
        st.error(f"Merge Error: {str(e)}")
        return None

# --- 1. DATA INGESTION ---
st.sidebar.header("1. Data Ingestion")
uploaded_files = st.sidebar.file_uploader("Upload PV & Weather Files", type=["csv", "xlsx"], accept_multiple_files=True)
df = None

if len(uploaded_files) >= 2:
    df = process_uploaded_files(uploaded_files[0], uploaded_files[1])
    
    if df is not None:
        if 'TempOut' in df.columns:
            df = df[df['TempOut'] <= 50]
        if 'SolarRad' in df.columns:
            df = df[df['SolarRad'] > 5] 
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['P_GEN'])

if df is not None:
    # --- 2. SITE ISOLATION ---
    st.sidebar.header("2. Site Isolation & Configuration")
    
    sub_col = 'Substation' if 'Substation' in df.columns else ('Substation_ID' if 'Substation_ID' in df.columns else None)
    if sub_col:
        sites = df[sub_col].unique().tolist()
        selected_site = st.sidebar.selectbox("Isolate Specific Substation:", sites)
        df = df[df[sub_col] == selected_site].copy()
        st.sidebar.success(f"Isolated: {selected_site}")
        
    df = df[~df.index.duplicated(keep='first')]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = st.sidebar.selectbox("Target Variable", numeric_cols, index=numeric_cols.index('P_GEN') if 'P_GEN' in numeric_cols else 0)
    
    if is_engineer:
        # --- Config (ENGINEER ONLY) ---
        if st.sidebar.button("Auto-Select Physics Features"):
            auto_feats, leaks = auto_select_features(df, target)
            st.session_state['selected_features'] = auto_feats
            st.sidebar.success(f"Selected {len(auto_feats)} drivers")
            
        features = st.sidebar.multiselect("Input Features", numeric_cols, default=st.session_state['selected_features'])
        selected_models = st.sidebar.multiselect("Algorithms", ["XGBoost", "RandomForest", "HistGradientBoosting", "LightGBM", "CatBoost"], default=["XGBoost", "HistGradientBoosting", "LightGBM"])
    else:
        features, _ = auto_select_features(df, target)
        selected_models = ["XGBoost", "HistGradientBoosting", "LightGBM"]

    # --- 3. EDA SECTION (Visible to both roles) ---
    st.header(f"Site Analysis: {selected_site if sub_col else 'Current Data'}")
    output_dir = "eda_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    eda_col1, eda_col2 = st.columns(2)
    if 'SolarRad' in df.columns and 'TempOut' in df.columns:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        scatter = ax1.scatter(df['SolarRad'], df[target], c=df['TempOut'], cmap='coolwarm', alpha=0.5)
        plt.colorbar(scatter, ax=ax1, label='Temperature (°C)')
        ax1.set_title(f"Physics Check: {target} vs Solar Radiation", fontsize=12)
        ax1.set_xlabel("Solar Radiation (W/m²)")
        ax1.set_ylabel(f"{target} (kW)")
        plt.tight_layout()
        eda_col1.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(df[target], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_title(f"Statistical Distribution: {target}", fontsize=12)
    ax2.set_xlabel(f"{target} (kW)")
    ax2.set_ylabel("Frequency")
    plt.tight_layout()
    eda_col2.pyplot(fig2)

    st.markdown("---")

    # --- 4. PHASE 1: TRAINING BLOCK ---
    if st.sidebar.button("Run Pipeline"):
        if not features:
            st.error("Feature selection required.")
        else:
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target])
            with st.spinner('Training and Tuning AI Pipeline...'):
                leaderboard, winner = run_autonomous_selection(df, features, target, selected_models)
                
                X = df[features]
                y = df[target].clip(lower=0) 
                split_idx = int(len(X) * 0.8)
                X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
                X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
                
                try:
                    tuned_model = optimise_winner(winner['model'], X_train, y_train)
                    used_model = tuned_model
                    final_preds = tuned_model.predict(X_test)
                except Exception as e:
                    used_model = winner['model']
                    final_preds = winner['model'].predict(X_test)

                # Save to the session state
                st.session_state.update({
                    'leaderboard': leaderboard, 'winner': winner, 'used_model': used_model,
                    'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test,
                    'full_y': df[target].clip(lower=0), 'final_preds': final_preds,
                    'final_r2': r2_score(y_test, final_preds), 'final_mae': mean_absolute_error(y_test, final_preds),
                    'features': features, 'df_full': df, 'training_done': True
                })

    # --- 5. POST-TRAINING DASHBOARD ---
    if st.session_state.get('training_done'):
        # --- ML Metrics (ENGINEER ONLY) ---
        if is_engineer:
            with st.container(border=True):
                st.subheader("Phase 1: Machine Learning Engineering Metrics")
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.write("**Model Tournament Leaderboard**")
                    st.dataframe(st.session_state['leaderboard'], use_container_width=True)
                with c2:
                    st.write("**Winning Configuration**")
                    st.success(f"{st.session_state['winner']['name']} (Tuned)")
                    st.metric("Optimized R² Score", f"{st.session_state['final_r2']:.4f}")
                    st.metric("Mean Absolute Error (MAE)", f"{st.session_state['final_mae']:.4f} kW")

        # --- High-Level Validation (BOTH ACTORS) ---
        st.subheader("Global Model Validation")
        c3, c4 = st.columns(2)
        with c3:
            fig_val, ax_val = plt.subplots(figsize=(8, 4))
            ax_val.plot(st.session_state['X_test'].index, st.session_state['y_test'], label='Actual PV', color='blue', alpha=0.5)
            ax_val.plot(st.session_state['X_test'].index, st.session_state['final_preds'], label='AI Prediction', color='red', linestyle='--', alpha=0.8)
            ax_val.set_title("Actual vs. Predicted Generation")
            ax_val.set_ylabel("Power (kW)")
            ax_val.legend()
            st.pyplot(fig_val)
        with c4:
            explainer, shap_values, X_sample = get_shap_explainer(st.session_state['used_model'], st.session_state['df_full'][st.session_state['features']])
            fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title("Global Physics Impact")
            st.pyplot(fig_shap)

        st.markdown("---")

        # --- 6. PHASE 2: HYBRID ANOMALY DETECTION ---
        st.subheader("Phase 2: Hybrid Anomaly Analysis")
        X_test, y_test, full_y = st.session_state['X_test'], st.session_state['y_test'], st.session_state['full_y']
        
        if 'SolarRad' in X_test.columns:
            efficiency_ratio = X_test['SolarRad'] / (y_test + 0.1)
            top_20 = efficiency_ratio.sort_values(ascending=False).head(20)
            anomaly_options = {f"{ts} | Actual: {y_test.loc[ts].iloc[0] if hasattr(y_test.loc[ts], '__len__') else y_test.loc[ts]:.2f}kW": ts for ts, _ in top_20.items()}
            
            selected_label = st.selectbox("Select Anomaly to Investigate:", list(anomaly_options.keys()))
            selected_ts = anomaly_options[selected_label]
            
            if st.button("Run Hybrid Diagnostic"):
                with st.spinner("Processing Hybrid Diagnostic Pipeline..."):
                    actual_val = y_test.loc[selected_ts].iloc[0] if hasattr(y_test.loc[selected_ts], '__len__') else y_test.loc[selected_ts]
                    row = X_test.loc[[selected_ts]]
                    ml_pred = st.session_state['used_model'].predict(row)[0]
                    
                    loc_res = full_y.index.get_loc(selected_ts)
                    hist_idx = loc_res.start if isinstance(loc_res, slice) else (np.where(loc_res)[0][0] if isinstance(loc_res, np.ndarray) else loc_res)
                    history = full_y.iloc[max(0, hist_idx-100):hist_idx]
                    arima_pred = run_hybrid_arima(history)

                    # --- Diagnostic Summary (BOTH ACTORS) ---
                    with st.container(border=True):
                        st.markdown("### Diagnostic Baselines")
                        c_a, c_b, c_c = st.columns(3)
                        c_a.metric("Actual Power (Sensor)", f"{actual_val:.2f} kW")
                        c_b.metric("ML Expected (Weather Physics)", f"{ml_pred:.2f} kW", f"{ml_pred - actual_val:.2f}", delta_color="inverse")
                        c_c.metric("ARIMA Expected (Historical Routine)", f"{arima_pred:.2f} kW" if arima_pred else "N/A", f"{arima_pred - actual_val:.2f}" if arima_pred else "", delta_color="inverse")

                    # --- Deep XAI & ARIMA Metrics (ENGINEER ONLY) ---
                    if is_engineer:
                        with st.expander("View Technical Analysis", expanded=True):

                            st.markdown("**Autonomous XAI Tournament Details**")
                            start_l = time.time()
                            explainer_lime = lime.lime_tabular.LimeTabularExplainer(
                                st.session_state['X_train'].values, 
                                feature_names=st.session_state['features'], 
                                mode='regression'
                            )
                            exp = explainer_lime.explain_instance(
                                row.values[0], 
                                lambda x: st.session_state['used_model'].predict(pd.DataFrame(x, columns=st.session_state['features']))
                            )
                            
                            start_s = time.time()
                            sh_val = shap.Explainer(st.session_state['used_model'].predict, shap.sample(st.session_state['X_train'], 100))(row)
                            
                            l_time  = time.time() - start_l
                            l_error = abs(ml_pred - exp.local_pred[0]) # LIME error
                            s_time  = time.time() - start_s
                            s_error = abs(ml_pred - (sh_val.base_values[0] + np.sum(sh_val.values[0]))) # SHAP error

                            l_rank = (1.5 * l_error) + (0.5 * l_time) # LIME rank
                            s_rank = (1.5 * s_error) + (0.5 * s_time) # SHAP rank

                            m1, m2, m3 = st.columns(3)
                            m1.metric("LIME Rank", f"{l_rank:.4f}", f"Err: {l_error:.4f} | {l_time:.2f}s")
                            m2.metric("SHAP Rank", f"{s_rank:.4f}", f"Err: {s_error:.4f} | {s_time:.2f}s")
                            
                            # DISPLAY THE WINNING GRAPH
                            if s_rank <= l_rank:
                                winner_name = "SHAP"
                                m3.success("Winner: SHAP (Mathematical Reliability)")
                                fig_s, ax_s = plt.subplots()
                                shap.plots.waterfall(sh_val[0], show=False) # Local SHAP graph
                                st.pyplot(fig_s)
                                best_contribs = sorted(list(zip(st.session_state['features'], sh_val.values[0])), key=lambda x: abs(x[1]), reverse=True)[:5]
                            else:
                                winner_name = "LIME"
                                m3.success("Winner: LIME (Execution Speed)")
                                st.pyplot(exp.as_pyplot_figure()) # Local LIME graph
                                best_contribs = exp.as_list()[:5]
                    else:
                        sh_val = shap.Explainer(st.session_state['used_model'].predict, shap.sample(st.session_state['X_train'], 100))(row)
                        best_contribs = sorted(list(zip(st.session_state['features'], sh_val.values[0])), key=lambda x: abs(x[1]), reverse=True)[:5]
                        winner_name = "SHAP"

                    # --- LLM Diagnostic Report (BOTH ACTORS) ---
                    with st.container(border=True):
                        st.markdown("### Final LLM Diagnostic Report")
                        
                        # Full raw data row for this timestamp to give the LLM context
                        full_sensor_row = st.session_state['df_full'].loc[selected_ts]
                        if isinstance(full_sensor_row, pd.DataFrame):
                            full_sensor_row = full_sensor_row.iloc[0]

                        report = generate_diagnostic_report(
                            prediction=ml_pred,
                            method_name=winner_name,
                            contributions=best_contribs,
                            anomaly_context=f"Actual power {actual_val:.2f}kW. ML Physics expected {ml_pred:.2f}kW. ARIMA History expected {arima_pred:.2f}kW.",
                            sensor_data=full_sensor_row
                        )
                        st.info(report)
else:
    st.info("Upload your PV and Weather files to begin.")