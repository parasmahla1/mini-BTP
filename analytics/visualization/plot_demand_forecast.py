import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_demand_forecast(historical_data, forecast, product_id=None, retailer_id=None):
    """
    Create visualization of demand forecasts.
    
    Args:
        historical_data: DataFrame with historical demand data
        forecast: DataFrame with forecasted values
        product_id: Optional product ID to filter the data
        retailer_id: Optional retailer ID to filter the data
    
    Returns:
        Matplotlib figure object
    """
    plt.figure(figsize=(12, 6))
    
    # Debug information
    print("Historical DataFrame columns:", historical_data.columns.tolist())
    print("Forecast DataFrame columns:", forecast.columns.tolist())
    
    # Filter data if product_id is provided
    if product_id is not None and 'product_id' in historical_data.columns:
        historical_data = historical_data[historical_data['product_id'] == product_id]
    
    # Filter data if retailer_id is provided
    if retailer_id is not None and 'retailer_id' in historical_data.columns:
        historical_data = historical_data[historical_data['retailer_id'] == retailer_id]
    
    # Plot historical data (last 30 days or all if less)
    days_to_show = min(30, len(historical_data))
    
    plt.plot(historical_data['date'].iloc[-days_to_show:], 
             historical_data['demand'].iloc[-days_to_show:], 
             'b-', marker='o', label='Historical Demand')
    
    # Check column names in forecast_data and use the correct one
    # Common column names for forecasts
    forecast_col_options = ['forecast', 'forecasted_demand', 'prediction', 'demand_forecast', 'value']
    
    forecast_col = None
    for col in forecast_col_options:
        if col in forecast.columns:
            forecast_col = col
            print(f"Found forecast column: '{col}'")
            break
    
    # If none of the expected columns exist, use the first numeric column
    if forecast_col is None:
        for col in forecast.columns:
            if col != 'date' and pd.api.types.is_numeric_dtype(forecast[col]):
                forecast_col = col
                print(f"Using column '{col}' for forecast values")
                break
    
    # Plot forecast data
    if forecast_col:
        plt.plot(forecast['date'], forecast[forecast_col], 
                'r-', marker='x', label='Forecasted Demand')
        
        # Plot confidence intervals if available
        if 'lower_bound' in forecast.columns and 'upper_bound' in forecast.columns:
            plt.fill_between(
                forecast['date'],
                forecast['lower_bound'],
                forecast['upper_bound'],
                color='r', alpha=0.2, label='95% Confidence Interval'
            )
    else:
        print("WARNING: No suitable forecast column found in forecast data")
        print("Available columns:", forecast.columns.tolist())
    
    # Add title and labels
    title = 'Demand Forecast'
    if product_id:
        title += f' for Product {product_id}'
    if retailer_id:
        title += f' at Retailer {retailer_id}'
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt