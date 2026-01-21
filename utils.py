# src/utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_expenses(df: pd.DataFrame, title: str = "Expenses Over Time") -> None:
    """
    Plot historical or forecasted expenses.
    
    Args:
        df: DataFrame with 'date' and 'expense' or 'predicted_expense'.
        title: Plot title
    """
    plt.figure(figsize=(10, 5))
    sns.lineplot(x='date', y=df.columns[1], data=df, marker='o')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Expense")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def merge_historical_and_forecast(historical_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine historical and forecasted expenses for visualization.
    
    Args:
        historical_df: DataFrame with 'date' and 'expense'
        forecast_df: DataFrame with 'date' and 'predicted_expense'
    
    Returns:
        Combined DataFrame with columns: 'date', 'expense', 'predicted_expense'
    """
    combined_df = pd.merge(
        historical_df,
        forecast_df,
        on='date',
        how='outer'
    ).sort_values('date').reset_index(drop=True)
    return combined_df

def convert_freq_to_string(freq: str) -> str:
    """
    Convert frequency code to human-readable string.
    
    Args:
        freq: 'M' for monthly, 'Q' for quarterly
    
    Returns:
        String description
    """
    if freq == 'M':
        return "Monthly"
    elif freq == 'Q':
        return "Quarterly"
    else:
        return freq

def safe_float(value, default=0.0):
    """
    Convert a value to float, return default if conversion fails.
    """
    try:
        return float(value)
    except Exception:
        return default
