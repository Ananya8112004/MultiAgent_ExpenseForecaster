import streamlit as st
import json
import requests
import time
from datetime import datetime
import pandas as pd

# --- Configuration ---
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
MAX_RETRIES = 5

# **MODIFICATION HERE: Hardcoded API Key**
# ----------------------------------------------------------------------
# IMPORTANT: Replace "YOUR_HARDCODED_API_KEY_HERE" with your actual Gemini API key (e.g., "xyz").
# ----------------------------------------------------------------------
HARDCODED_API_KEY = "AIzaSyC2hj1SjuJQcCoxcTXJYW9e6l3ZXF3tlMg"


# --- Helper Functions ---

def create_prediction_prompt(historical_data, prediction_period):
    """Generates the detailed prompt for the Gemini model."""

    # Define the expected JSON schema for a structured, reliable output
    # The model will be instructed to return ONLY this JSON object.
    
    # NOTE: The schema definition is crucial for reliable parsing.
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "predicted_total_expense": {
                "type": "NUMBER",
                "description": "The overall predicted total expense for the next period."
            },
            "prediction_period": {
                "type": "STRING",
                "description": f"The predicted period, clearly stating if it is the next quarter or month based on user input '{prediction_period}'."
            },
            "expense_breakdown": {
                "type": "ARRAY",
                "description": "A list of predicted expenses categorized by category.",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "category": { "type": "STRING" },
                        "predicted_amount": { "type": "NUMBER" },
                        "justification": { "type": "STRING", "description": "Brief reason for this category's prediction (e.g., 'Seasonal increase for holidays', 'Stable baseline', or 'Projected increase due to known factor')." }
                    },
                    "propertyOrdering": ["category", "predicted_amount", "justification"]
                }
            },
            "key_insights": {
                "type": "STRING",
                "description": "A concise, single paragraph (100-150 words) summarizing the key historical patterns, seasonal trends, and anomalous data points found in the provided data. This should justify the overall prediction."
            }
        },
        "propertyOrdering": ["predicted_total_expense", "prediction_period", "expense_breakdown", "key_insights"]
    }

    prompt = f"""
    You are a world-class Financial Forecasting Agent. Your task is to analyze historical expense data, identify seasonal patterns, trends, and anomalies, and predict the expenses for the next **{prediction_period}**.

    **Historical Data (CSV format):**
    {historical_data}

    **Analysis and Prediction Requirements:**
    1.  Analyze the provided data which is typically structured as 'Date,Category,Amount'.
    2.  Calculate the total predicted expense for the next **{prediction_period}**.
    3.  Provide a detailed breakdown of the predicted expenses by category.
    4.  Offer a single paragraph of key insights justifying your prediction based on identified trends (e.g., Q4 holiday spending spikes, consistent monthly rent, summer travel increases, etc.).
    5.  The final output MUST strictly adhere to the requested JSON schema. Do not include any text, markdown formatting, or explanations outside the JSON block.
    """
    
    return prompt, response_schema

def call_gemini_api(api_key, prompt, response_schema):
    """Calls the Gemini API with exponential backoff."""
    headers = {'Content-Type': 'application/json'}
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "config": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }

    for i in range(MAX_RETRIES):
        try:
            # Use fetch directly without the API key, as the environment provides it during runtime
            if api_key:
                url_with_key = f"{API_URL}?key={api_key}"
            else:
                url_with_key = API_URL
                
            response = requests.post(url_with_key, headers=headers, json=payload)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            
            if result.get('candidates'):
                json_text = result['candidates'][0]['content']['parts'][0]['text']
                # The model is forced to output JSON, but it might be wrapped in markdown backticks.
                if json_text.startswith("```json"):
                    json_text = json_text.strip().replace("```json", "").replace("```", "").strip()
                return json.loads(json_text)
            
            # If no candidates, but no HTTP error, it might be a safety block
            st.error("API Error: The request was blocked or returned an empty response. Check the prompt content.")
            return None

        except requests.exceptions.RequestException as e:
            st.warning(f"Attempt {i+1} failed due to connection error or API issue: {e}")
            if i < MAX_RETRIES - 1:
                wait_time = 2 ** i
                time.sleep(wait_time)
            else:
                st.error("All retries failed. Please check your API key and connection.")
                return None
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON response from the API. This often means the model output was not valid JSON. Error: {e}")
            st.code(json_text) # Show the raw text output for debugging
            return None
    return None

# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="AI Expense Predictor")

st.markdown("""
# 🧠 AI Financial Forecaster
Predict your next period's expenses using the Gemini API.
The AI analyzes your historical data for seasonal patterns and trends to provide a reliable forecast.
""")

with st.sidebar:
    # MODIFICATION: Show hardcoded status instead of an input field.
    st.header("1. API Configuration")
    if HARDCODED_API_KEY == "YOUR_HARDCODED_API_KEY_HERE":
        st.warning("API Key is currently a placeholder. Please edit the code to add your key.")
    else:
        st.success("API Key is hardcoded in the script.")
    
    st.header("2. Prediction Period")
    prediction_period = st.selectbox(
        "Select Prediction Horizon",
        ("Next Quarter (3 months)", "Next Month (1 month)"),
        key="period_select"
    )

st.header("3. Historical Expense Data Input")
st.info("Paste your historical expense data below. **Use CSV format** with column headers: `Date`, `Category`, `Amount`.")

example_data = f"""Date,Category,Amount
2024-01-05,Rent,2000
2024-01-10,Groceries,350
2024-01-15,Travel,150
2024-02-05,Rent,2000
2024-02-12,Groceries,400
2024-03-05,Rent,2000
2024-03-20,Groceries,300
2024-03-25,Travel,2500
2024-04-05,Rent,2000
2024-04-10,Groceries,375
2024-05-05,Rent,2000
2024-05-15,Groceries,420
2024-06-05,Rent,2000
2024-06-25,Entertainment,1000
2024-07-05,Rent,2000
2024-07-10,Groceries,360
2024-08-05,Rent,2000
2024-08-20,Groceries,450
2024-09-05,Rent,2000
2024-09-30,Travel,500
2024-10-05,Rent,2000
2024-10-15,Groceries,500
2024-11-05,Rent,2000
2024-11-20,Shopping,1500
2024-12-05,Rent,2000
2024-12-10,Groceries,600
2024-12-25,Gifts,1200"""


historical_data = st.text_area(
    "Historical Data (CSV)", 
    value=example_data,
    height=300
)

# Button to trigger prediction
if st.button("Generate Forecast", type="primary"):
    # MODIFICATION: Check the hardcoded key placeholder
    if HARDCODED_API_KEY == "YOUR_HARDCODED_API_KEY_HERE":
        st.error("Please replace 'YOUR_HARDCODED_API_KEY_HERE' in the script with your actual Gemini API Key.")
    elif not historical_data.strip():
        st.error("Please provide historical expense data.")
    else:
        # Create prompt and schema
        prompt, response_schema = create_prediction_prompt(historical_data, prediction_period)
        
        # Display a spinner while waiting for the API call
        with st.spinner(f"Analyzing data and generating forecast for the {prediction_period}..."):
            # MODIFICATION: Pass the hardcoded key
            prediction_data = call_gemini_api(HARDCODED_API_KEY, prompt, response_schema)

        if prediction_data:
            st.success(f"Forecast Generated for {prediction_data.get('prediction_period', prediction_period)}!")
            
            # --- Display Results ---
            
            st.subheader("💰 Predicted Total Expense")
            total_expense = prediction_data.get('predicted_total_expense', 'N/A')
            st.metric(
                label=f"Forecast for {prediction_data.get('prediction_period', prediction_period)}", 
                value=f"${total_expense:,.2f}" if isinstance(total_expense, (int, float)) else str(total_expense)
            )

            st.subheader("📊 Expense Breakdown")
            breakdown = prediction_data.get('expense_breakdown', [])

            if breakdown:
                df_breakdown = pd.DataFrame(breakdown)
                # Ensure the predicted_amount column is numeric for charting
                df_breakdown['predicted_amount'] = pd.to_numeric(df_breakdown['predicted_amount'], errors='coerce')
                
                # Display as a chart
                st.bar_chart(df_breakdown, x='category', y='predicted_amount', color="#4a69bd", use_container_width=True)
                
                # Display as a table with justifications
                st.dataframe(
                    df_breakdown.rename(columns={
                        'category': 'Category',
                        'predicted_amount': 'Predicted Amount ($)',
                        'justification': 'Forecasting Justification'
                    }),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Predicted Amount ($)": st.column_config.NumberColumn(format="%.2f")
                    }
                )
            else:
                st.warning("No detailed category breakdown was returned by the model.")

            st.subheader("📈 Key Insights from the Analysis")
            st.markdown(f"*{prediction_data.get('key_insights', 'The model did not provide specific insights.')}*")
            
            # Add an image tag to illustrate the concept of time series forecasting.
            # This helps users understand the underlying method.
            st.markdown("")

        else:
            st.error("Prediction failed. Please check the API key, data format, or see the console for error details.")