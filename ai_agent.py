# src/ai_agent.py

import os
import pandas as pd
import requests
from datetime import timedelta
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
API_URL = f"https://api.gemini.ai/v1/flash/{GEMINI_MODEL}"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}" if API_KEY else "",
    "Content-Type": "application/json",
}

def format_prompt(df: pd.DataFrame, periods: int, freq: str) -> str:
    """Format historical expense data into a prompt for Gemini."""
    data_str = "\n".join([f"{row['date'].strftime('%Y-%m-%d')}: {row['expense']}" for _, row in df.iterrows()])
    prompt = (
        f"Given the historical expense data below aggregated {freq}ly, "
        f"predict the expense values for the next {periods} {freq} periods. "
        f"Output only the dates and predicted expenses in the format YYYY-MM-DD: amount.\n\n"
        f"Historical data:\n{data_str}\n\n"
        f"Predictions:"
    )
    return prompt

def parse_response(text: str) -> pd.DataFrame:
    """Parse the Gemini API response text into a DataFrame."""
    rows = []
    for line in text.strip().split("\n"):
        if ':' in line:
            date_str, value_str = line.split(":", 1)
            try:
                date = pd.to_datetime(date_str.strip())
                expense = float(value_str.strip())
                rows.append({"date": date, "predicted_expense": expense})
            except Exception:
                continue
    return pd.DataFrame(rows)

def get_ai_forecast(df: pd.DataFrame, periods: int, freq: str) -> pd.DataFrame:
    """Get forecast predictions from Gemini, with offline fallback."""
    if API_KEY is None:
        print("GEMINI_API_KEY not found. Using offline fallback.")
        return offline_forecast(df, periods, freq)

    prompt = format_prompt(df, periods, freq)

    payload = {
        "model": GEMINI_MODEL,
        "prompt": prompt,
        "temperature": 0.3,
        "max_tokens": 150,
        "top_p": 1,
        "n": 1,
        "stop": ["\n\n"],
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        generated_text = result.get("choices", [{}])[0].get("text", "").strip()
        forecast_df = parse_response(generated_text)

        if forecast_df.empty:
            print("Gemini returned empty response. Using incremental fallback.")
            return offline_forecast(df, periods, freq)

        return forecast_df

    except Exception as e:
        print(f"Gemini API unreachable or failed: {e}")
        return offline_forecast(df, periods, freq)

def offline_forecast(df: pd.DataFrame, periods: int, freq: str) -> pd.DataFrame:
    """Fallback forecast using historical mean and incremental dates."""
    last_date = df['date'].max()
    freq_map = {'M': pd.DateOffset(months=1), 'Q': pd.DateOffset(months=3)}
    date_offset = freq_map.get(freq, pd.DateOffset(months=1))
    return pd.DataFrame({
        "date": [last_date + date_offset * (i + 1) for i in range(periods)],
        "predicted_expense": [df['expense'].mean()] * periods
    })
