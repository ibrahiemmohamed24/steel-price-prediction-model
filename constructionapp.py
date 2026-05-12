import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Steel Price Forecasting System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ASSET LOADING ---
@st.cache_resource
def load_resources():
    """
    Load the trained models and the feature list saved from the notebook.
    """
    try:
        rf_model = joblib.load('rf_model_v3.pkl')
        xgb_model = joblib.load('xgb_model_v3.pkl')
        features = joblib.load('feature_cols.pkl')
        return rf_model, xgb_model, features
    except FileNotFoundError as e:
        st.error(f"Error: Required model files not found. {e}")
        return None, None, None

rf_model, xgb_model, feature_names = load_resources()

# --- APP HEADER ---
st.title("Steel Price Prediction Dashboard")
st.subheader("Construction Material Cost Forecasting - Egypt Market")
st.write("Input the current economic indicators below to generate a price forecast for Rebar (16mm).")

# --- SIDEBAR: INPUT PARAMETERS ---
st.sidebar.header("Economic Indicators")

def get_user_inputs():
    """
    Captures user input from the sidebar and returns a structured DataFrame.
    """
    # Grouping inputs for better UI organization
    st.sidebar.subheader("Currency & Global Markets")
    usd_avg = st.sidebar.number_input("USD to EGP Exchange Rate", value=48.0, step=0.1)
    iron_ore = st.sidebar.number_input("Global Iron Ore (USD/Ton)", value=110.0, step=0.5)
    brent_oil = st.sidebar.number_input("Brent Oil (USD/Barrel)", value=80.0, step=0.5)
    nat_gas = st.sidebar.number_input("Natural Gas (USD/MMBtu)", value=2.5, step=0.1)
    
    st.sidebar.subheader("Macroeconomics")
    inflation = st.sidebar.number_input("Inflation Rate (%)", value=35.0, step=0.1)
    interest = st.sidebar.number_input("Interest Rate (%)", value=27.0, step=0.1)
    
    st.sidebar.subheader("Lagged Indicators & Indices")
    iron_lag = st.sidebar.number_input("Last Month Iron Ore (USD)", value=108.0)
    oil_lag = st.sidebar.number_input("Last Month Brent Oil (USD)", value=82.0)
    local_idx = st.sidebar.number_input("Local Price/USD Index", value=800.0)

    # Creating a dictionary with exact keys used during training
    input_data = {
        'USD_Average': usd_avg,
        'Global_Iron_Ore_USD': iron_ore,
        'Brent_Oil_USD': brent_oil,
        'Natural_Gas_USD': nat_gas,
        'Inflation': inflation,
        'Interest_Rate': interest,
        'Global_Iron_Lagged': iron_lag,
        'Oil_Lagged': oil_lag,
        'Local_Index_USD': local_idx
    }
    return pd.DataFrame([input_data])

user_data = get_user_inputs()

# --- PREDICTION LOGIC ---
if st.button("Generate Forecast"):
    if rf_model and xgb_model:
        # Ensure feature order matches the training phase
        ordered_data = user_data[feature_names]
        
        # Individual Model Predictions
        rf_prediction = rf_model.predict(ordered_data)[0]
        xgb_prediction = xgb_model.predict(ordered_data)[0]
        
        # Weighted Ensemble (50% RF / 50% XGB)
        final_forecast = (0.5 * rf_prediction) + (0.5 * xgb_prediction)
        
        # Display Results
        st.markdown("---")
        main_col, info_col = st.columns([2, 1])
        
        with main_col:
            st.metric(
                label="Predicted Steel Price (EGP / Ton)", 
                value=f"{final_forecast:,.2f}"
            )
            st.write("Prediction generated using a Hybrid Ensemble Model (RandomForest & XGBoost).")
            
        with info_col:
            st.info("Model performance metrics suggest a MAPE of ~4.5% based on recent test datasets.")
    else:
        st.warning("Models are not initialized. Check your .pkl files.")

# --- FOOTER ---
st.markdown("---")
st.caption("Forecasts are for informational purposes only. Local market conditions may vary.")