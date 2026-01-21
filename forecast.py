# src/forecast.py

import pandas as pd
from src.seasonality import detect_seasonality, adjust_for_seasonality
from src.trend import detect_trend, adjust_for_trend
from src.ai_agent import get_ai_forecast

def forecast_expenses(
    df: pd.DataFrame,
    periods: int = 1,
    freq: str = 'M'
) -> pd.DataFrame:
    """
    Forecast future expenses for the given number of periods.
    
    Args:
        df: Historical expenses dataframe with 'date' and 'expense' columns.
        periods: Number of future periods to predict.
        freq: Frequency for aggregation ('M' monthly, 'Q' quarterly).
    
    Returns:
        pd.DataFrame with forecasted 'date' and 'predicted_expense'.
    """
    # Step 1: Detect seasonality and trend
    seasonal_pattern = detect_seasonality(df, freq=freq)
    trend_pattern = detect_trend(df, freq=freq)
    
    # Step 2: Call AI agent for base forecast (pass historical data)
    base_forecast = get_ai_forecast(df, periods, freq)
    
    # Step 3: Adjust forecast for seasonality and trend
    adjusted_forecast = adjust_for_seasonality(base_forecast, seasonal_pattern)
    adjusted_forecast = adjust_for_trend(adjusted_forecast, trend_pattern)
    
    return adjusted_forecast
