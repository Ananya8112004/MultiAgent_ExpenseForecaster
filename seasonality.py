# src/seasonality.py

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def detect_seasonality(df: pd.DataFrame, freq: str = 'M') -> pd.Series:
    """
    Detect seasonality component in the historical expense data.
    
    Args:
        df: DataFrame with 'date' and 'expense' columns.
        freq: Frequency for resampling/aggregation ('M' for monthly, 'Q' for quarterly).
        
    Returns:
        pandas Series representing the seasonal component indexed by period.
    """
    # Resample if needed
    df = df.set_index('date')
    ts = df['expense'].resample(freq).sum()

    # Handle if length too short for decomposition
    if len(ts) < 2 * 12:  # less than 2 years of monthly data approx
        # Seasonality detection unreliable
        return pd.Series([1] * len(ts), index=ts.index)
    
    decomposition = seasonal_decompose(ts, model='multiplicative', period=12, extrapolate_trend='freq')
    seasonal = decomposition.seasonal
    
    # Normalize seasonal component to be around 1 (multiplicative)
    normalized_seasonal = seasonal / seasonal.mean()
    
    return normalized_seasonal

def adjust_for_seasonality(forecast_df: pd.DataFrame, seasonal_pattern: pd.Series) -> pd.DataFrame:
    """
    Adjust forecasted expenses using the detected seasonal pattern.
    
    Args:
        forecast_df: DataFrame with 'date' and 'predicted_expense'.
        seasonal_pattern: Series with seasonal multipliers indexed by date.
        
    Returns:
        Adjusted forecast DataFrame with same structure.
    """
    adjusted_forecast = forecast_df.copy()
    adjusted_expenses = []
    
    for forecast_date in adjusted_forecast['date']:
        # Find the seasonal factor corresponding to the same month/quarter
        if forecast_date in seasonal_pattern.index:
            factor = seasonal_pattern.loc[forecast_date]
        else:
            # Match by month or quarter if exact date missing
            factor = _find_seasonal_factor(seasonal_pattern, forecast_date)
        adjusted_expenses.append(adjusted_forecast.loc[adjusted_forecast['date'] == forecast_date, 'predicted_expense'].values[0] * factor)
    
    adjusted_forecast['predicted_expense'] = adjusted_expenses
    return adjusted_forecast

def _find_seasonal_factor(seasonal_pattern: pd.Series, date: pd.Timestamp) -> float:
    """
    Helper to find seasonal factor by matching month or quarter if exact date is missing.
    """
    # Try month match
    for idx in seasonal_pattern.index:
        if idx.month == date.month:
            return seasonal_pattern.loc[idx]
    # If quarterly, match quarter
    for idx in seasonal_pattern.index:
        if hasattr(idx, 'quarter') and idx.quarter == date.quarter:
            return seasonal_pattern.loc[idx]
    # Fallback
    return 1.0
