import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    # Load housing data
    housing_data = pd.read_csv('https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
    
    # Try different encodings for the demographic data
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            demographic_data = pd.read_csv('https://www2.census.gov/programs-surveys/popest/datasets/2020-2022/metro/totals/cbsa-est2022.csv', encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Unable to read the demographic data with any of the attempted encodings.")
    
    # Process housing data
    housing_data['RegionName'] = housing_data['RegionName'].str.lower()
    
    # Identify date columns (assuming they are in YYYY-MM-DD format)
    date_columns = [col for col in housing_data.columns if '-' in col]
    
    # Melt the dataframe
    id_vars = [col for col in housing_data.columns if col not in date_columns]
    housing_data = housing_data.melt(id_vars=id_vars, var_name='Date', value_name='Price')
    housing_data['Date'] = pd.to_datetime(housing_data['Date'])
    
    # Process demographic data
    demographic_data['NAME'] = demographic_data['NAME'].str.lower()
    demographic_data = demographic_data.rename(columns={'NAME': 'RegionName', 'POPESTIMATE2022': 'Population'})
    
    # Merge datasets
    combined_df = pd.merge(housing_data, demographic_data[['RegionName', 'Population']], 
                           on='RegionName', how='left')
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    combined_df = pd.DataFrame(imputer.fit_transform(combined_df), columns=combined_df.columns)
    
    # Separate features and target
    X = combined_df.drop(['Price', 'Date', 'RegionName', 'Metro', 'CountyName', 'RegionID', 'SizeRank'], axis=1, errors='ignore')
    y = combined_df['Price']
    
    # Handle categorical variables
    X = pd.get_dummies(X, columns=['State'])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, X.columns, scaler

def train_model(X, y):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return best_model, X_test, y_test, y_pred, mse, r2, grid_search.best_params_

def load_model_and_scaler():
    model = joblib.load('housing_price_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

def predict_price(features, model, scaler, feature_names):
    # Create a DataFrame with the correct column names
    features_df = pd.DataFrame([features], columns=feature_names)
    
    # Fill missing columns with 0 (for one-hot encoded columns that might be missing)
    for col in model.feature_names_in_:
        if col not in features_df.columns:
            features_df[col] = 0
    
    # Reorder columns to match the order used during training
    features_df = features_df[model.feature_names_in_]
    
    # Scale the features
    features_scaled = scaler.transform(features_df)
    
    return model.predict(features_scaled)[0]

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Housing Prices')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.close()

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig('residual_plot.png')
    plt.close()

if __name__ == "__main__":
    # Load and preprocess data
    X, y, feature_names, scaler = load_and_preprocess_data()
    
    # Train model
    model, X_test, y_test, y_pred, mse, r2, best_params = train_model(X, y)
    
    # Print results
    print(f"Mean squared error: {mse:.2f}")
    print(f"R-squared score: {r2:.2f}")
    print(f"Best parameters: {best_params}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Create visualizations
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    
    # Save model and scaler
    joblib.dump(model, 'housing_price_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    print("\nModel and scaler saved. Visualizations created.")
    
    # Example usage
    new_house_features = np.array([1500, 3, 2, 2000, 5000, 95123])  # Example values
    predicted_price = predict_price(new_house_features, model, scaler, feature_names)
    print(f"\nPredicted price for new house: ${predicted_price:.2f}")
    
    # Print feature names for reference
    print("\nFeature names:")
    print(feature_names)
