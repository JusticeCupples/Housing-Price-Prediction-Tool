# Housing Price Prediction Tool

This project is a Python-based tool that predicts housing prices for different states in the USA using historical data and simple projections.

## Features

- Fetches 20 years of historical housing price data from the Federal Reserve Economic Data (FRED)
- Displays housing price trends for the past 4 years and projects 2 years into the future
- Creates a graph showing the price trends and saves it as an image
- Provides a simple command-line interface for user input

## Files

1. `housing_price_model.py`: The main script containing all the functions for data fetching, processing, and visualization.
2. `predict_price.py`: An identical copy of `housing_price_model.py`, possibly used for testing or as a backup.
3. `feature_names.joblib`: A joblib file containing feature names used in the prediction model.
4. `housing_price_model.joblib`: A joblib file containing the trained prediction model.
5. `scaler.joblib`: A joblib file containing the scaler used to normalize input data.
6. `state_populations.db`: A SQLite database containing state population data.

## Key Functions

- `fetch_housing_data(state)`: Fetches housing data for a given state from FRED.
- `predict_prices_for_past_and_future(state, df, model, scaler, feature_names, median_price)`: Processes historical data and makes future price predictions.
- `create_graph(state, dates, prices)`: Creates and saves a graph of housing prices.

## Usage

Run the `housing_price_model.py` script: python housing_price_model.py

When prompted, enter the two-letter abbreviation for the state you want to analyze (e.g., CA for California).

## Output

The script will print:

- The latest House Price Index for the state
- The year-over-year price change
- The current median home price
- Predicted prices for the state (annually)
- The price range (min to max)

It will also save a graph as `price_prediction.png` in the current directory.

## Dependencies

- joblib
- numpy
- matplotlib
- pandas
- fredapi
- tkinter

## Note

This tool uses a simple projection method for future prices and should not be used for actual financial decisions. The median home prices are placeholder values and should be updated with real data for accurate results.

## Author

- Justice Cupples - https://github.com/JusticeCupples
