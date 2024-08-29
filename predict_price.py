import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import simpledialog
import logging
from logging.handlers import RotatingFileHandler
import sys
import requests
import json
import pandas as pd
from fredapi import Fred

# Set up logging
log_file = 'predict_price_error.log'
handler = RotatingFileHandler(log_file, maxBytes=100000, backupCount=3)
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[handler])

def fetch_median_home_price(state):
    # This is a placeholder function. In a real-world scenario, you'd fetch this data from a reliable source.
    # For now, we'll use some example values.
    median_prices = {
        'CA': 700000,
        'NY': 650000,
        'TX': 300000,
        # Add more states as needed
    }
    return median_prices.get(state, 300000)  # Default to 300000 if state not found

def fetch_housing_data(state):
    fred = Fred(api_key='fd8e2c9448bee26ed2a16103ef744064')
    
    # FRED series ID for All-Transactions House Price Index by State
    series_id = f'{state}STHPI'
    
    # Fetch data for the last 20 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*20)
    
    data = fred.get_series(series_id, start_date, end_date)
    
    # Convert to DataFrame and reset index
    df = pd.DataFrame(data).reset_index()
    df.columns = ['date', 'price_index']
    
    # Calculate year-over-year price change
    df['price_change'] = df['price_index'].pct_change(periods=4)  # Assuming quarterly data
    
    # Get the most recent price index and year-over-year change
    latest_index = df['price_index'].iloc[-1]
    latest_change = df['price_change'].iloc[-1]
    
    median_price = fetch_median_home_price(state)
    return df, latest_index, latest_change, median_price

def predict_price_for_state(state, price_index, price_change, model, scaler, feature_names, other_features=None):
    # For now, let's just return the price index as is
    return price_index

def load_joblib_file(filename):
    try:
        return joblib.load(filename)
    except FileNotFoundError:
        parent_dir = os.path.dirname(os.getcwd())
        file_path = os.path.join(parent_dir, filename)
        if os.path.exists(file_path):
            return joblib.load(file_path)
        else:
            raise FileNotFoundError(f"Could not find {filename} in current or parent directory")

def predict_prices_for_past_and_future(state, df, model, scaler, feature_names, median_price):
    dates = []
    prices = []
    current_date = datetime.now()

    # Calculate the price per index point
    price_per_index = median_price / df['price_index'].iloc[-1]

    # Past 4 years (use actual data)
    past_4_years = df[df['date'] >= current_date - timedelta(days=365*4)]
    for _, row in past_4_years.iterrows():
        dates.append(row['date'])
        prices.append(row['price_index'] * price_per_index)

    # Future 2 years (simple projection)
    last_index = df['price_index'].iloc[-1]
    last_change = df['price_change'].iloc[-1]
    for i in range(1, 25):  # 24 months (2 years)
        future_date = current_date + timedelta(days=30*i)
        future_index = last_index * (1 + last_change) ** (i/4)  # Assuming quarterly data
        dates.append(future_date)
        prices.append(future_index * price_per_index)

    return dates, prices

def create_graph(state, dates, prices):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices)
    plt.title(f'Housing Prices for {state} (4 Years Past and 2 Years Future)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set y-axis range
    min_price = min(prices)
    max_price = max(prices)
    plt.ylim(min_price * 0.95, max_price * 1.05)

    # Add vertical line for current date
    current_date = datetime.now()
    plt.axvline(x=current_date, color='r', linestyle='--', label='Current Date')

    # Add annotations for min and max prices
    min_date = dates[prices.index(min(prices))]
    max_date = dates[prices.index(max(prices))]

    plt.annotate(f'Min: ${min(prices):,.0f}', xy=(min_date, min(prices)), xytext=(10, 10),
                 textcoords='offset points', ha='left', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

    plt.annotate(f'Max: ${max(prices):,.0f}', xy=(max_date, max(prices)), xytext=(10, -10),
                 textcoords='offset points', ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

    plt.legend()
    plt.tight_layout()

    # Remove the existing file if it exists
    if os.path.exists('price_prediction.png'):
        os.remove('price_prediction.png')

    plt.savefig('price_prediction.png')
    plt.close()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    try:
        state = simpledialog.askstring("Input", "Enter the state abbreviation (e.g., CA for California):").upper()
        if not state:
            raise ValueError("State input is required")
        
        df, latest_index, latest_change, median_price = fetch_housing_data(state)
        print(f"Latest House Price Index for {state}: {latest_index:.2f}")
        print(f"Year-over-year price change: {latest_change:.2%}")
        print(f"Current median home price: ${median_price:,.0f}")
        
        model = load_joblib_file('housing_price_model.joblib')
        scaler = load_joblib_file('scaler.joblib')
        feature_names = load_joblib_file('feature_names.joblib')
        
        dates, prices = predict_prices_for_past_and_future(state, df, model, scaler, feature_names, median_price)
        create_graph(state, dates, prices)
        
        print(f"\nPredicted prices for {state}:")
        for date, price in zip(dates[::12], prices[::12]):  # Print every 12 months
            print(f"{date.strftime('%Y-%m-%d')}: ${price:,.0f}")
        
        print(f"\nPrice range: ${min(prices):,.0f} to ${max(prices):,.0f}")
        
        print(f"\nGraph saved as 'price_prediction.png' in the current directory.")
    except ValueError as ve:
        logging.error(f"Value Error: {str(ve)}", exc_info=True)
        print(f"Value Error: {str(ve)}")
    except FileNotFoundError as fnf:
        logging.error(f"File Not Found Error: {str(fnf)}", exc_info=True)
        print(f"File Not Found Error: {str(fnf)}")
    except requests.RequestException as re:
        logging.error(f"API Request Error: {str(re)}", exc_info=True)
        print(f"API Request Error: {str(re)}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        print(f"An unexpected error occurred. Please check the log file: {log_file}")
        print(f"Error details: {str(e)}")
