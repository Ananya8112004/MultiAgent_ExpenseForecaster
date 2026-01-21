# src/trend.py

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

def detect_trend(df: pd.DataFrame, freq: str = 'M') -> pd.Series:
    """
    Detect trend component in the historical expense data.
    
    Args:
        df: DataFrame with 'date' and 'expense' columns.
        freq: Frequency for resampling ('M' for monthly, 'Q' for quarterly)
    
    Returns:
        pandas Series representing trend values indexed by date.
    """
    df = df.set_index('date')
    ts = df['expense'].resample(freq).sum()

    # Handle short series
    if len(ts) < 3:
        # Too short to detect trend reliably, return series of ones
        return pd.Series([1] * len(ts), index=ts.index)
    
    # Use STL decomposition to extract trend
    stl = STL(ts, seasonal=13 if freq=='M' else 3, robust=True)
    result = stl.fit()
    trend = result.trend
    
    # Normalize trend relative to last value to scale future forecasts
    normalized_trend = trend / trend.iloc[-1]
    
    return normalized_trend

def adjust_for_trend(forecast_df: pd.DataFrame, trend_pattern: pd.Series) -> pd.DataFrame:
    """
    Adjust forecasted expenses using the detected trend component.
    
    Args:
        forecast_df: DataFrame with 'date' and 'predicted_expense'.
        trend_pattern: Series representing trend, indexed by historical dates.
    
    Returns:
        Adjusted forecast DataFrame with same structure.
    """
    adjusted_forecast = forecast_df.copy()
    
    last_trend_value = trend_pattern.iloc[-1] if not trend_pattern.empty else 1.0
    
    adjusted_expenses = []
    for i, forecast_date in enumerate(adjusted_forecast['date']):
        # Assume linear trend continuation
        factor = 1 + (i + 1) * (last_trend_value - 1) / max(len(trend_pattern), 1)
        adjusted_expense = adjusted_forecast.loc[i, 'predicted_expense'] * factor
        adjusted_expenses.append(adjusted_expense)
    
    adjusted_forecast['predicted_expense'] = adjusted_expenses
    return adjusted_forecast
