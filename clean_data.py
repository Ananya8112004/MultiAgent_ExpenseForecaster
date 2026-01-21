# src/clean_data.py

import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load expense data from CSV.
    Expected columns: 'date' and 'expense' (or similar).
    """
    df = pd.read_csv(filepath)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data:
    - Parse dates
    - Drop rows with missing values
    - Rename columns if necessary
    """
    # Ensure 'date' column exists and parse to datetime
    if 'date' not in df.columns:
        raise ValueError("CSV must have a 'date' column")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Ensure 'expense' column exists
    if 'expense' not in df.columns:
        # Try to infer expense column if possible or raise error
        possible_expense_cols = [col for col in df.columns if 'expense' in col.lower()]
        if possible_expense_cols:
            df.rename(columns={possible_expense_cols[0]: 'expense'}, inplace=True)
        else:
            raise ValueError("CSV must have an 'expense' column")

    # Drop rows with missing dates or expenses
    df = df.dropna(subset=['date', 'expense'])

    # Convert expense to numeric
    df['expense'] = pd.to_numeric(df['expense'], errors='coerce')
    df = df.dropna(subset=['expense'])

    return df

def aggregate_expenses(df: pd.DataFrame, freq: str = 'M') -> pd.DataFrame:
    """
    Aggregate expenses by given frequency.
    freq: 'M' for monthly, 'Q' for quarterly, etc.
    Returns dataframe with 'date' and 'expense' aggregated.
    """
    # Set date as index for resampling
    df = df.set_index('date')
    agg_df = df['expense'].resample(freq).sum().reset_index()
    return agg_df
