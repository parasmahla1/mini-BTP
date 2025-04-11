"""
Evaluation utilities for supply chain predictive models.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_forecast_model(actual, forecast, date_col='date', actual_col='demand', forecast_col='forecasted_demand'):
    """
    Evaluate demand forecasting model performance.
    
    Args:
        actual (pd.DataFrame): DataFrame with actual values
        forecast (pd.DataFrame): DataFrame with forecasted values
        date_col (str): Column name for date
        actual_col (str): Column name for actual values
        forecast_col (str): Column name for forecasted values
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Ensure date columns are datetime
    actual[date_col] = pd.to_datetime(actual[date_col])
    forecast[date_col] = pd.to_datetime(forecast[date_col])
    
    # Merge actual and forecast data
    merged = pd.merge(
        actual,
        forecast,
        on=[date_col, 'product_id', 'retailer_id'],
        how='inner'
    )
    
    if len(merged) == 0:
        return {
            'MAE': np.nan,
            'RMSE': np.nan,
            'MAPE': np.nan,
            'R2': np.nan
        }
    
    # Calculate metrics
    y_true = merged[actual_col]
    y_pred = merged[forecast_col]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    non_zero_mask = (y_true != 0)
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    
    # R2 score
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def evaluate_lead_time_model(actual, predicted):
    """
    Evaluate lead time prediction model.
    
    Args:
        actual (np.array): Array of actual lead times
        predicted (np.array): Array of predicted lead times
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # MPE (Mean Percentage Error) to check for bias
    non_zero_mask = (actual != 0)
    mpe = np.mean((predicted[non_zero_mask] - actual[non_zero_mask]) / actual[non_zero_mask]) * 100
    
    # Calculate accuracy within different thresholds
    accuracy_1day = np.mean(np.abs(predicted - actual) <= 1)
    accuracy_2day = np.mean(np.abs(predicted - actual) <= 2)
    accuracy_3day = np.mean(np.abs(predicted - actual) <= 3)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MPE': mpe,
        'Accuracy_1day': accuracy_1day,
        'Accuracy_2day': accuracy_2day,
        'Accuracy_3day': accuracy_3day
    }


def evaluate_inventory_policy(demand, inventory_levels, reorder_points, order_quantities):
    """
    Evaluate inventory policy performance.
    
    Args:
        demand (np.array): Array of actual demand values
        inventory_levels (np.array): Array of inventory levels
        reorder_points (np.array): Array of reorder points
        order_quantities (np.array): Array of order quantities
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Stockout rate
    stockouts = (demand > inventory_levels).sum()
    stockout_rate = stockouts / len(demand) if len(demand) > 0 else 0
    
    # Average inventory level
    avg_inventory = np.mean(inventory_levels)
    
    # Inventory turns
    total_demand = np.sum(demand)
    inventory_turns = total_demand / avg_inventory if avg_inventory > 0 else 0
    
    # Service level
    fulfilled_demand = np.minimum(demand, inventory_levels).sum()
    service_level = fulfilled_demand / total_demand if total_demand > 0 else 0
    
    return {
        'Stockout_Rate': stockout_rate,
        'Avg_Inventory': avg_inventory,
        'Inventory_Turns': inventory_turns,
        'Service_Level': service_level
    }


def evaluate_disruption_predictions(actual, predicted, threshold=0.5):
    """
    Evaluate disruption prediction model.
    
    Args:
        actual (np.array): Array of actual disruption indicators (0/1)
        predicted (np.array): Array of predicted disruption probabilities
        threshold (float): Probability threshold for classification
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Convert probabilities to binary predictions
    binary_predictions = (predicted >= threshold).astype(int)
    
    # True positives, false positives, etc.
    tp = ((binary_predictions == 1) & (actual == 1)).sum()
    fp = ((binary_predictions == 1) & (actual == 0)).sum()
    tn = ((binary_predictions == 0) & (actual == 0)).sum()
    fn = ((binary_predictions == 0) & (actual == 1)).sum()
    
    # Metrics
    accuracy = (tp + tn) / len(actual) if len(actual) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1
    }