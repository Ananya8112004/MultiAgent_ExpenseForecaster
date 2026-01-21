# streamlit_app.py


from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

# Optional: check if API key is loaded
if os.getenv("GEMINI_API_KEY") is None:
    st.warning("GEMINI_API_KEY not found in environment variables!")



import streamlit as st
import pandas as pd
from src.clean_data import load_data, clean_data, aggregate_expenses
from src.forecast import forecast_expenses
from src.utils import plot_expenses, merge_historical_and_forecast, convert_freq_to_string

st.set_page_config(page_title="Expense Forecaster", layout="wide")

st.title("💰 Expense Forecaster AI")
st.markdown(
    """
    Upload your historical expense data (CSV) and forecast future expenses using AI.
    The AI agent uses historical trends and seasonal patterns to predict future expenses.
    """
)

# --- Sidebar options ---
st.sidebar.header("Forecast Settings")

freq_option = st.sidebar.selectbox("Select Forecast Frequency", options=['M', 'Q'], index=0)
periods = st.sidebar.number_input("Forecast Periods", min_value=1, max_value=24, value=3, step=1)

# --- Data upload ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    df_clean = clean_data(df_raw)
    df_agg = aggregate_expenses(df_clean, freq=freq_option)

    st.subheader("Historical Expenses")
    st.dataframe(df_agg)
    
    # --- Forecasting ---
    st.subheader("Forecasted Expenses")
    forecast_df = forecast_expenses(df_agg, periods=periods, freq=freq_option)
    
    st.dataframe(forecast_df)
    
    # --- Combine for visualization ---
    combined_df = merge_historical_and_forecast(df_agg.rename(columns={'expense':'expense'}), forecast_df)
    
    st.subheader("Historical + Forecast Visualization")
    plot_expenses(combined_df, title=f"{convert_freq_to_string(freq_option)} Expenses Forecast")
    
    # --- Download option ---
    csv = combined_df.to_csv(index=False)
    st.download_button(
        label="Download Combined Data as CSV",
        data=csv,
        file_name="expense_forecast.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file to start forecasting. You can use the sample CSV provided.")
