"""
Comprehensive evaluation utilities for supply chain predictive models.
This module provides robust metrics calculation, visualizations, and cross-validation
for various supply chain models including demand forecasting, lead time prediction,
inventory optimization, and disruption prediction.
"""
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    precision_recall_curve, roc_curve, auc, confusion_matrix,
    precision_score, recall_score, f1_score
)
from scipy.stats import pearsonr, spearmanr


def evaluate_forecast_model(
    actual: pd.DataFrame, 
    forecast: pd.DataFrame, 
    date_col: str = 'date', 
    actual_col: str = 'demand', 
    forecast_col: str = 'forecast',
    group_cols: Optional[List[str]] = None,
    visualize: bool = False,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Evaluate demand forecasting model performance with comprehensive metrics.
    
    Args:
        actual: DataFrame with actual values
        forecast: DataFrame with forecasted values
        date_col: Column name for date
        actual_col: Column name for actual values
        forecast_col: Column name for forecasted values
        group_cols: Optional list of columns to group by (e.g., ['product_id', 'retailer_id'])
        visualize: Whether to generate visualizations
        output_dir: Directory to save visualizations (if None, plots are displayed)
        
    Returns:
        Dictionary of evaluation metrics including overall and per-group metrics
    """
    # Input validation
    if actual.empty or forecast.empty:
        warnings.warn("Empty dataframe provided for evaluation")
        return _empty_forecast_metrics()
    
    # Ensure required columns exist
    required_cols = [date_col]
    if group_cols:
        required_cols.extend(group_cols)
    
    for df, name, col in [(actual, "actual", actual_col), (forecast, "forecast", forecast_col)]:
        if col not in df.columns:
            warnings.warn(f"Column {col} not found in {name} dataframe")
            return _empty_forecast_metrics()
        
        for req_col in required_cols:
            if req_col not in df.columns:
                warnings.warn(f"Required column {req_col} not found in {name} dataframe")
                return _empty_forecast_metrics()
    
    # Ensure date columns are datetime
    actual[date_col] = pd.to_datetime(actual[date_col])
    forecast[date_col] = pd.to_datetime(forecast[date_col])
    
    # Prepare merge columns
    merge_cols = [date_col]
    if group_cols:
        merge_cols.extend(group_cols)
    
    # Merge actual and forecast data
    try:
        merged = pd.merge(
            actual,
            forecast,
            on=merge_cols,
            how='inner'
        )
    except Exception as e:
        warnings.warn(f"Error merging actual and forecast data: {e}")
        return _empty_forecast_metrics()
    
    if len(merged) == 0:
        warnings.warn("No matching data points after merging actual and forecast")
        return _empty_forecast_metrics()
    
    # Calculate overall metrics
    y_true = merged[actual_col]
    y_pred = merged[forecast_col]
    
    overall_metrics = _calculate_forecast_metrics(y_true, y_pred)
    results = {"overall": overall_metrics}
    
    # Calculate per-group metrics if specified
    if group_cols:
        group_metrics = {}
        for name, group in merged.groupby(group_cols):
            # Handle single and multi-column group names
            if isinstance(name, tuple):
                group_name = "_".join([str(n) for n in name])
            else:
                group_name = str(name)
                
            group_true = group[actual_col]
            group_pred = group[forecast_col]
            
            if len(group_true) > 0:
                group_metrics[group_name] = _calculate_forecast_metrics(group_true, group_pred)
        
        results["by_group"] = group_metrics
    
    # Generate visualizations if requested
    if visualize:
        visualizations = _visualize_forecast_results(
            merged, 
            actual_col, 
            forecast_col, 
            date_col, 
            group_cols,
            output_dir
        )
        results["visualizations"] = visualizations
    
    return results


def _empty_forecast_metrics():
    """Return empty metrics dictionary for error cases."""
    return {
        'overall': {
            'MAE': np.nan,
            'RMSE': np.nan,
            'MAPE': np.nan,
            'SMAPE': np.nan,
            'R2': np.nan,
            'MBE': np.nan,
            'TheilU': np.nan,
            'MASE': np.nan,
            'sample_size': 0
        }
    }


def _calculate_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate comprehensive forecast evaluation metrics."""
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Bias Error (to detect systematic bias)
    mbe = np.mean(y_pred - y_true)
    
    # Percentage-based errors (handle zeros carefully)
    non_zero_mask = (y_true != 0)
    
    # MAPE (Mean Absolute Percentage Error)
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / 
                              y_true[non_zero_mask])) * 100
    else:
        mape = np.nan
    
    # SMAPE (Symmetric Mean Absolute Percentage Error)
    denominator = np.abs(y_true) + np.abs(y_pred)
    valid_mask = (denominator != 0)
    if np.any(valid_mask):
        smape = np.mean(2.0 * np.abs(y_pred[valid_mask] - y_true[valid_mask]) / 
                        denominator[valid_mask]) * 100
    else:
        smape = np.nan
    
    # Theil's U statistic (closer to 0 is better)
    if np.sum(y_true**2) > 0 and np.sum(y_pred**2) > 0:
        theil_u = np.sqrt(np.sum((y_pred - y_true)**2)) / \
                 (np.sqrt(np.sum(y_true**2)) + np.sqrt(np.sum(y_pred**2)))
    else:
        theil_u = np.nan
    
    # MASE (Mean Absolute Scaled Error) - using naive seasonal (1-period) forecast as baseline
    if len(y_true) > 1:
        # Create naive forecast (shift by 1)
        naive_forecast = np.concatenate([[np.nan], y_true[:-1]])
        naive_errors = np.abs(y_true[1:] - naive_forecast[1:])
        naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else np.nan
        
        if naive_mae > 0 and not np.isnan(naive_mae):
            mase = mae / naive_mae
        else:
            mase = np.nan
    else:
        mase = np.nan
    
    # Correlation metrics
    if len(y_true) > 1:
        try:
            pearson_r, _ = pearsonr(y_true, y_pred)
            spearman_r, _ = spearmanr(y_true, y_pred)
        except Exception:
            pearson_r, spearman_r = np.nan, np.nan
    else:
        pearson_r, spearman_r = np.nan, np.nan
    
    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'SMAPE': float(smape),
        'R2': float(r2),
        'MBE': float(mbe),
        'TheilU': float(theil_u),
        'MASE': float(mase),
        'PearsonR': float(pearson_r),
        'SpearmanR': float(spearman_r),
        'sample_size': int(len(y_true))
    }


def _visualize_forecast_results(
    merged_data: pd.DataFrame, 
    actual_col: str, 
    forecast_col: str, 
    date_col: str,
    group_cols: Optional[List[str]],
    output_dir: Optional[str]
) -> Dict:
    """Generate visualizations for forecast evaluation."""
    visualization_paths = {}
    
    # Convert output_dir to str to avoid type issues
    output_path = output_dir if output_dir else None
    
    # Actual vs Predicted Scatter Plot
    plt.figure(figsize=(10, 6))
    max_val = max(merged_data[actual_col].max(), merged_data[forecast_col].max())
    min_val = min(merged_data[actual_col].min(), merged_data[forecast_col].min())
    plt.scatter(merged_data[actual_col], merged_data[forecast_col], alpha=0.5)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        import os
        scatter_path = os.path.join(output_path, 'forecast_scatter.png')
        plt.savefig(scatter_path)
        visualization_paths['scatter'] = scatter_path
    else:
        plt.show()
    plt.close()
    
    # Residuals Plot
    plt.figure(figsize=(10, 6))
    residuals = merged_data[forecast_col] - merged_data[actual_col]
    plt.scatter(merged_data[actual_col], residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Actual Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        residuals_path = os.path.join(output_path, 'forecast_residuals.png')
        plt.savefig(residuals_path)
        visualization_paths['residuals'] = residuals_path
    else:
        plt.show()
    plt.close()
    
    # Distribution of Residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        hist_path = os.path.join(output_path, 'forecast_residuals_hist.png')
        plt.savefig(hist_path)
        visualization_paths['residuals_hist'] = hist_path
    else:
        plt.show()
    plt.close()
    
    # Time Series Plot (if data has date and sorted)
    if date_col in merged_data.columns:
        if group_cols and len(group_cols) > 0:
            # Time series for each group (limit to top N groups by data volume)
            top_groups = merged_data.groupby(group_cols).size().sort_values(ascending=False).head(5).index
            
            plt.figure(figsize=(12, 8))
            for i, group_val in enumerate(top_groups):
                group_data = merged_data.loc[merged_data[group_cols[0]] == group_val].copy()
                group_data = group_data.sort_values(by=date_col)
                plt.plot(group_data[date_col], group_data[actual_col], 
                         marker='o', label=f'Actual - {group_val}', linestyle='-', alpha=0.7)
                plt.plot(group_data[date_col], group_data[forecast_col], 
                         marker='x', label=f'Forecast - {group_val}', linestyle='--', alpha=0.7)
            
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.title('Actual vs Forecast Time Series by Group')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if output_path:
                ts_path = os.path.join(output_path, 'forecast_time_series.png')
                plt.savefig(ts_path)
                visualization_paths['time_series'] = ts_path
            else:
                plt.show()
            plt.close()
        else:
            # Single time series
            plt.figure(figsize=(12, 6))
            sorted_data = merged_data.sort_values(by=date_col)
            plt.plot(sorted_data[date_col], sorted_data[actual_col], 
                     marker='o', label='Actual', linestyle='-', alpha=0.7)
            plt.plot(sorted_data[date_col], sorted_data[forecast_col], 
                     marker='x', label='Forecast', linestyle='--', alpha=0.7)
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.title('Actual vs Forecast Time Series')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if output_path:
                ts_path = os.path.join(output_path, 'forecast_time_series.png')
                plt.savefig(ts_path)
                visualization_paths['time_series'] = ts_path
            else:
                plt.show()
            plt.close()
    
    return visualization_paths


def evaluate_lead_time_model(
    actual: np.ndarray, 
    predicted: np.ndarray,
    entity_ids: Optional[np.ndarray] = None,
    material_ids: Optional[np.ndarray] = None,
    visualize: bool = False,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Evaluate lead time prediction model with enhanced metrics.
    
    Args:
        actual: Array of actual lead times
        predicted: Array of predicted lead times
        entity_ids: Optional array of entity IDs for group-based evaluation
        material_ids: Optional array of material IDs for group-based evaluation
        visualize: Whether to generate visualizations
        output_dir: Directory to save visualizations (if None, plots are displayed)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Input validation
    if len(actual) == 0 or len(predicted) == 0:
        warnings.warn("Empty arrays provided for evaluation")
        return _empty_lead_time_metrics()
    
    if len(actual) != len(predicted):
        warnings.warn(f"Length mismatch: actual ({len(actual)}) vs predicted ({len(predicted)})")
        return _empty_lead_time_metrics()
    
    # Calculate overall metrics
    overall_metrics = _calculate_lead_time_metrics(actual, predicted)
    results = {"overall": overall_metrics}
    
    # Group-based evaluation
    if entity_ids is not None and len(entity_ids) == len(actual):
        entity_metrics = {}
        for entity_id in np.unique(entity_ids):
            entity_mask = (entity_ids == entity_id)
            if np.sum(entity_mask) > 0:
                entity_actual = actual[entity_mask]
                entity_predicted = predicted[entity_mask]
                entity_metrics[str(entity_id)] = _calculate_lead_time_metrics(entity_actual, entity_predicted)
        
        results["by_entity"] = entity_metrics
    
    if material_ids is not None and len(material_ids) == len(actual):
        material_metrics = {}
        for material_id in np.unique(material_ids):
            material_mask = (material_ids == material_id)
            if np.sum(material_mask) > 0:
                material_actual = actual[material_mask]
                material_predicted = predicted[material_mask]
                material_metrics[str(material_id)] = _calculate_lead_time_metrics(material_actual, material_predicted)
        
        results["by_material"] = material_metrics
    
    # Generate visualizations if requested
    if visualize:
        visualizations = _visualize_lead_time_results(
            actual, 
            predicted, 
            entity_ids, 
            material_ids,
            output_dir
        )
        results["visualizations"] = visualizations
    
    return results


def _empty_lead_time_metrics():
    """Return empty metrics dictionary for error cases."""
    return {
        'overall': {
            'MAE': np.nan,
            'RMSE': np.nan,
            'MPE': np.nan,
            'MAPE': np.nan,
            'R2': np.nan,
            'MBE': np.nan,
            'Accuracy_1day': np.nan,
            'Accuracy_2day': np.nan,
            'Accuracy_3day': np.nan,
            'sample_size': 0
        }
    }


def _calculate_lead_time_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict:
    """Calculate comprehensive lead time evaluation metrics."""
    # Basic metrics
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    
    # R2 score
    r2 = r2_score(actual, predicted)
    
    # Bias metrics
    mbe = np.mean(predicted - actual)  # Mean Bias Error
    
    # Percentage errors (avoiding division by zero)
    non_zero_mask = (actual != 0)
    
    # MPE (Mean Percentage Error)
    if np.any(non_zero_mask):
        mpe = np.mean((predicted[non_zero_mask] - actual[non_zero_mask]) / 
                      actual[non_zero_mask]) * 100
        mape = np.mean(np.abs((predicted[non_zero_mask] - actual[non_zero_mask]) / 
                              actual[non_zero_mask])) * 100
    else:
        mpe = np.nan
        mape = np.nan
    
    # Accuracy within thresholds
    accuracy_1day = np.mean(np.abs(predicted - actual) <= 1)
    accuracy_2day = np.mean(np.abs(predicted - actual) <= 2)
    accuracy_3day = np.mean(np.abs(predicted - actual) <= 3)
    
    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MPE': float(mpe),
        'MAPE': float(mape),
        'R2': float(r2),
        'MBE': float(mbe),
        'Accuracy_1day': float(accuracy_1day),
        'Accuracy_2day': float(accuracy_2day),
        'Accuracy_3day': float(accuracy_3day),
        'sample_size': int(len(actual))
    }


def _visualize_lead_time_results(
    actual: np.ndarray, 
    predicted: np.ndarray,
    entity_ids: Optional[np.ndarray],
    material_ids: Optional[np.ndarray],
    output_dir: Optional[str]
) -> Dict:
    """Generate visualizations for lead time evaluation."""
    visualization_paths = {}
    
    # Convert output_dir to str to avoid type issues
    output_path = output_dir if output_dir else None
    
    # Actual vs Predicted Scatter Plot
    plt.figure(figsize=(10, 6))
    max_val = max(np.max(actual), np.max(predicted))
    min_val = min(np.min(actual), np.min(predicted))
    plt.scatter(actual, predicted, alpha=0.5)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual Lead Time (days)')
    plt.ylabel('Predicted Lead Time (days)')
    plt.title('Actual vs. Predicted Lead Times')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        import os
        scatter_path = os.path.join(output_path, 'lead_time_scatter.png')
        plt.savefig(scatter_path)
        visualization_paths['scatter'] = scatter_path
    else:
        plt.show()
    plt.close()
    
    # Error Distribution
    plt.figure(figsize=(10, 6))
    errors = predicted - actual
    sns.histplot(errors, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error (days)')
    plt.ylabel('Frequency')
    plt.title('Lead Time Prediction Error Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        error_path = os.path.join(output_path, 'lead_time_error_dist.png')
        plt.savefig(error_path)
        visualization_paths['error_distribution'] = error_path
    else:
        plt.show()
    plt.close()
    
    # Group-based analysis if available
    if entity_ids is not None and len(np.unique(entity_ids)) <= 20:  # Limit to 20 entities for visibility
        # Entity-wise accuracy
        plt.figure(figsize=(12, 8))
        entity_accuracy = {}
        for entity_id in np.unique(entity_ids):
            entity_mask = (entity_ids == entity_id)
            if np.sum(entity_mask) >= 5:  # Require at least 5 data points
                entity_actual = actual[entity_mask]
                entity_predicted = predicted[entity_mask]
                entity_accuracy[entity_id] = np.mean(np.abs(entity_predicted - entity_actual) <= 1)
        
        if entity_accuracy:
            entities = list(entity_accuracy.keys())
            accuracies = list(entity_accuracy.values())
            y_pos = np.arange(len(entities))
            plt.barh(y_pos, accuracies)
            plt.yticks(y_pos, entities)
            plt.xlabel('1-Day Accuracy')
            plt.ylabel('Entity ID')
            plt.title('Lead Time Prediction Accuracy by Entity')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if output_path:
                entity_path = os.path.join(output_path, 'lead_time_entity_accuracy.png')
                plt.savefig(entity_path)
                visualization_paths['entity_accuracy'] = entity_path
            else:
                plt.show()
            plt.close()
    
    return visualization_paths


def evaluate_inventory_policy(
    demand: np.ndarray, 
    inventory_levels: np.ndarray, 
    reorder_points: np.ndarray, 
    order_quantities: np.ndarray,
    holding_cost_rate: float = 0.25,
    stockout_cost: float = 50.0,
    ordering_cost: float = 100.0,
    visualize: bool = False,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Evaluate inventory policy performance with comprehensive metrics.
    
    Args:
        demand: Array of actual demand values
        inventory_levels: Array of inventory levels
        reorder_points: Array of reorder points
        order_quantities: Array of order quantities
        holding_cost_rate: Annual holding cost as a fraction of item value
        stockout_cost: Cost per stockout event
        ordering_cost: Fixed cost per order
        visualize: Whether to generate visualizations
        output_dir: Directory to save visualizations (if None, plots are displayed)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Input validation
    if (len(demand) == 0 or len(inventory_levels) == 0 or 
        len(reorder_points) == 0 or len(order_quantities) == 0):
        warnings.warn("Empty arrays provided for evaluation")
        return _empty_inventory_metrics()
    
    # Ensure all arrays are of the same length
    if not (len(demand) == len(inventory_levels) == len(reorder_points) == len(order_quantities)):
        warnings.warn("Length mismatch among input arrays")
        return _empty_inventory_metrics()
    
    # Calculate metrics
    metrics = _calculate_inventory_metrics(
        demand, 
        inventory_levels, 
        reorder_points, 
        order_quantities,
        holding_cost_rate,
        stockout_cost,
        ordering_cost
    )
    results = {"metrics": metrics}
    
    # Generate visualizations if requested
    if visualize:
        visualizations = _visualize_inventory_results(
            demand, 
            inventory_levels, 
            reorder_points, 
            order_quantities,
            output_dir
        )
        results["visualizations"] = visualizations
    
    return results


def _empty_inventory_metrics():
    """Return empty metrics dictionary for error cases."""
    return {
        'Stockout_Rate': np.nan,
        'Avg_Inventory': np.nan,
        'Inventory_Turns': np.nan,
        'Service_Level': np.nan,
        'Fill_Rate': np.nan,
        'Ready_Rate': np.nan,
        'Total_Cost': np.nan,
        'Holding_Cost': np.nan,
        'Stockout_Cost': np.nan,
        'Ordering_Cost': np.nan,
        'Order_Frequency': np.nan,
        'sample_size': 0
    }


def _calculate_inventory_metrics(
    demand: np.ndarray, 
    inventory_levels: np.ndarray, 
    reorder_points: np.ndarray, 
    order_quantities: np.ndarray,
    holding_cost_rate: float,
    stockout_cost: float,
    ordering_cost: float
) -> Dict:
    """Calculate comprehensive inventory policy evaluation metrics."""
    # Basic metrics
    stockouts = (demand > inventory_levels)
    stockout_count = np.sum(stockouts)
    stockout_rate = stockout_count / len(demand) if len(demand) > 0 else np.nan
    
    # Average inventory level
    avg_inventory = np.mean(inventory_levels)
    
    # Inventory turns
    total_demand = np.sum(demand)
    inventory_turns = total_demand / avg_inventory if avg_inventory > 0 else np.nan
    
    # Service metrics
    # Fill Rate: percentage of demand satisfied immediately from stock
    fulfilled_demand = np.minimum(demand, inventory_levels).sum()
    fill_rate = fulfilled_demand / total_demand if total_demand > 0 else np.nan
    
    # Ready Rate: percentage of time with positive inventory
    ready_rate = np.mean(inventory_levels > 0)
    
    # Cost metrics
    # Holding cost: annual holding cost rate * average inventory
    holding_cost = holding_cost_rate * avg_inventory
    
    # Stockout cost: cost per stockout * number of stockouts
    stockout_cost_total = stockout_cost * stockout_count
    
    # Ordering cost: fixed cost per order * number of orders
    # Assuming an order is placed when inventory falls below reorder point
    order_placements = (inventory_levels <= reorder_points)
    order_count = np.sum(order_placements)
    order_frequency = order_count / len(inventory_levels) if len(inventory_levels) > 0 else np.nan
    ordering_cost_total = ordering_cost * order_count
    
    # Total cost
    total_cost = holding_cost + stockout_cost_total + ordering_cost_total
    
    return {
        'Stockout_Rate': float(stockout_rate),
        'Avg_Inventory': float(avg_inventory),
        'Inventory_Turns': float(inventory_turns),
        'Service_Level': float(1 - stockout_rate),
        'Fill_Rate': float(fill_rate),
        'Ready_Rate': float(ready_rate),
        'Total_Cost': float(total_cost),
        'Holding_Cost': float(holding_cost),
        'Stockout_Cost': float(stockout_cost_total),
        'Ordering_Cost': float(ordering_cost_total),
        'Order_Frequency': float(order_frequency),
        'sample_size': int(len(demand))
    }


def _visualize_inventory_results(
    demand: np.ndarray, 
    inventory_levels: np.ndarray, 
    reorder_points: np.ndarray, 
    order_quantities: np.ndarray,
    output_dir: Optional[str]
) -> Dict:
    """Generate visualizations for inventory policy evaluation."""
    visualization_paths = {}
    
    # Convert output_dir to str to avoid type issues
    output_path = output_dir if output_dir else None
    
    # Inventory vs Demand Time Series Plot
    plt.figure(figsize=(12, 6))
    time_periods = np.arange(len(demand))
    plt.plot(time_periods, inventory_levels, 'b-', label='Inventory Level')
    plt.plot(time_periods, demand, 'r-', label='Demand', alpha=0.7)
    plt.plot(time_periods, reorder_points, 'g--', label='Reorder Point')
    plt.xlabel('Time Period')
    plt.ylabel('Units')
    plt.title('Inventory Level vs Demand')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        import os
        ts_path = os.path.join(output_path, 'inventory_time_series.png')
        plt.savefig(ts_path)
        visualization_paths['time_series'] = ts_path
    else:
        plt.show()
    plt.close()
    
    # Stockout Analysis
    plt.figure(figsize=(10, 6))
    stockouts = (demand > inventory_levels)
    plt.scatter(time_periods, stockouts.astype(int), marker='x', color='r', alpha=0.7)
    plt.xlabel('Time Period')
    plt.ylabel('Stockout Event')
    plt.title('Stockout Events Over Time')
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        stockout_path = os.path.join(output_path, 'inventory_stockouts.png')
        plt.savefig(stockout_path)
        visualization_paths['stockouts'] = stockout_path
    else:
        plt.show()
    plt.close()
    
    # Order Placement Analysis
    plt.figure(figsize=(10, 6))
    orders = (inventory_levels <= reorder_points)
    order_sizes = orders.astype(float) * order_quantities
    order_mask = (order_sizes > 0)
    
    if np.any(order_mask):
        plt.scatter(time_periods[order_mask], order_sizes[order_mask], 
                   marker='^', color='g', alpha=0.7)
        plt.xlabel('Time Period')
        plt.ylabel('Order Size')
        plt.title('Order Placements and Sizes')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            order_path = os.path.join(output_path, 'inventory_orders.png')
            plt.savefig(order_path)
            visualization_paths['orders'] = order_path
        else:
            plt.show()
        plt.close()
    
    return visualization_paths


def evaluate_disruption_predictions(
    actual: np.ndarray, 
    predicted_probs: np.ndarray,
    threshold: float = 0.5,
    entity_ids: Optional[np.ndarray] = None,
    visualize: bool = False,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Evaluate disruption prediction model with comprehensive metrics.
    
    Args:
        actual: Array of actual disruption indicators (0/1)
        predicted_probs: Array of predicted disruption probabilities
        threshold: Probability threshold for binary classification
        entity_ids: Optional array of entity IDs for group-based evaluation
        visualize: Whether to generate visualizations
        output_dir: Directory to save visualizations (if None, plots are displayed)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Input validation
    if len(actual) == 0 or len(predicted_probs) == 0:
        warnings.warn("Empty arrays provided for evaluation")
        return _empty_disruption_metrics()
    
    if len(actual) != len(predicted_probs):
        warnings.warn(f"Length mismatch: actual ({len(actual)}) vs predicted ({len(predicted_probs)})")
        return _empty_disruption_metrics()
    
    # Calculate overall metrics
    binary_predictions = (predicted_probs >= threshold).astype(int)
    overall_metrics = _calculate_disruption_metrics(actual, binary_predictions, predicted_probs)
    
    # Calculate optimal threshold
    optimal_threshold, threshold_metrics = _calculate_optimal_threshold(actual, predicted_probs)
    
    results = {
        "overall": overall_metrics,
        "threshold_analysis": {
            "optimal_threshold": float(optimal_threshold),
            **threshold_metrics
        }
    }
    
    # Group-based evaluation
    if entity_ids is not None and len(entity_ids) == len(actual):
        entity_metrics = {}
        for entity_id in np.unique(entity_ids):
            entity_mask = (entity_ids == entity_id)
            if np.sum(entity_mask) > 0:
                entity_actual = actual[entity_mask]
                entity_predicted = binary_predictions[entity_mask]
                entity_probs = predicted_probs[entity_mask]
                entity_metrics[str(entity_id)] = _calculate_disruption_metrics(
                    entity_actual, entity_predicted, entity_probs)
        
        results["by_entity"] = entity_metrics
    
    # Generate visualizations if requested
    if visualize:
        visualizations = _visualize_disruption_results(
            actual, 
            predicted_probs, 
            binary_predictions,
            threshold,
            optimal_threshold,
            entity_ids,
            output_dir
        )
        results["visualizations"] = visualizations
    
    return results


def _empty_disruption_metrics():
    """Return empty metrics dictionary for error cases."""
    return {
        'Accuracy': np.nan,
        'Precision': np.nan,
        'Recall': np.nan,
        'F1_Score': np.nan,
        'AUC_ROC': np.nan,
        'AUC_PR': np.nan,
        'LogLoss': np.nan,
        'Brier_Score': np.nan,
        'sample_size': 0
    }


def _calculate_disruption_metrics(
    actual: np.ndarray, 
    binary_predictions: np.ndarray,
    predicted_probs: np.ndarray
) -> Dict:
    """Calculate comprehensive disruption prediction metrics."""
    # Basic classification metrics
    accuracy = np.mean(binary_predictions == actual)
    
    # Handle edge case of all negative or all positive predictions
    if np.all(binary_predictions == 0) or np.all(binary_predictions == 1):
        precision = np.nan
        recall = np.nan
        f1 = np.nan
    else:
        precision = precision_score(actual, binary_predictions, zero_division=0)
        recall = recall_score(actual, binary_predictions, zero_division=0)
        f1 = f1_score(actual, binary_predictions, zero_division=0)
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(actual, binary_predictions, labels=[0, 1]).ravel()
    
    # Calculate more advanced metrics
    try:
        # ROC AUC
        fpr, tpr, _ = roc_curve(actual, predicted_probs)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall AUC
        precision_curve, recall_curve, _ = precision_recall_curve(actual, predicted_probs)
        pr_auc = auc(recall_curve, precision_curve)
    except Exception:
        roc_auc = np.nan
        pr_auc = np.nan
    
    # Log loss (indicator of prediction confidence)
    # Clip probabilities to avoid log(0) issues
    eps = 1e-15
    clipped_probs = np.clip(predicted_probs, eps, 1 - eps)
    log_loss = -np.mean(actual * np.log(clipped_probs) + 
                       (1 - actual) * np.log(1 - clipped_probs))
    
    # Brier score (mean squared error of probabilities)
    brier_score = np.mean((predicted_probs - actual) ** 2)
    
    return {
        'Accuracy': float(accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1_Score': float(f1),
        'AUC_ROC': float(roc_auc),
        'AUC_PR': float(pr_auc),
        'LogLoss': float(log_loss),
        'Brier_Score': float(brier_score),
        'True_Positives': int(tp),
        'False_Positives': int(fp),
        'True_Negatives': int(tn),
        'False_Negatives': int(fn),
        'sample_size': int(len(actual))
    }


def _calculate_optimal_threshold(actual: np.ndarray, predicted_probs: np.ndarray) -> Tuple[float, Dict]:
    """Calculate optimal probability threshold for binary classification."""
    try:
        # Calculate precision, recall etc. at different thresholds
        precision_curve, recall_curve, thresholds_pr = precision_recall_curve(actual, predicted_probs)
        
        # Calculate F1 score at each threshold
        f1_scores = np.zeros_like(thresholds_pr)
        for i, threshold in enumerate(thresholds_pr):
            binary_preds = (predicted_probs >= threshold).astype(int)
            f1_scores[i] = f1_score(actual, binary_preds, zero_division=0)
        
        # Find threshold that maximizes F1 score
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds_pr[best_idx]
        
        return optimal_threshold, {
            'F1_Score': float(f1_scores[best_idx]),
            'Precision': float(precision_curve[best_idx]),
            'Recall': float(recall_curve[best_idx])
        }
    except Exception as e:
        warnings.warn(f"Error calculating optimal threshold: {e}")
        return 0.5, {
            'F1_Score': np.nan,
            'Precision': np.nan,
            'Recall': np.nan
        }


def _visualize_disruption_results(
    actual: np.ndarray, 
    predicted_probs: np.ndarray,
    binary_predictions: np.ndarray,
    threshold: float,
    optimal_threshold: float,
    entity_ids: Optional[np.ndarray],
    output_dir: Optional[str]
) -> Dict:
    """Generate visualizations for disruption prediction evaluation."""
    visualization_paths = {}
    
    # Convert output_dir to str to avoid type issues
    output_path = output_dir if output_dir else None
    
    # ROC Curve
    try:
        plt.figure(figsize=(8, 8))
        fpr, tpr, _ = roc_curve(actual, predicted_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Disruption Prediction')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            import os
            roc_path = os.path.join(output_path, 'disruption_roc_curve.png')
            plt.savefig(roc_path)
            visualization_paths['roc_curve'] = roc_path
        else:
            plt.show()
        plt.close()
    except Exception as e:
        warnings.warn(f"Error generating ROC curve: {e}")
    
    # Precision-Recall Curve
    try:
        plt.figure(figsize=(8, 8))
        precision_curve, recall_curve, _ = precision_recall_curve(actual, predicted_probs)
        pr_auc = auc(recall_curve, precision_curve)
        
        plt.plot(recall_curve, precision_curve, 'b-', label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.axhline(y=np.mean(actual), color='k', linestyle='--', 
                   label=f'Baseline (Prevalence = {np.mean(actual):.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for Disruption Prediction')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            pr_path = os.path.join(output_path, 'disruption_pr_curve.png')
            plt.savefig(pr_path)
            visualization_paths['pr_curve'] = pr_path
        else:
            plt.show()
        plt.close()
    except Exception as e:
        warnings.warn(f"Error generating PR curve: {e}")
    
    # Confusion Matrix Heatmap
    try:
        conf_matrix = confusion_matrix(actual, binary_predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Disruption', 'Disruption'],
                   yticklabels=['No Disruption', 'Disruption'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix (Threshold = {threshold:.2f})')
        plt.tight_layout()
        
        if output_path:
            cm_path = os.path.join(output_path, 'disruption_confusion_matrix.png')
            plt.savefig(cm_path)
            visualization_paths['confusion_matrix'] = cm_path
        else:
            plt.show()
        plt.close()
    except Exception as e:
        warnings.warn(f"Error generating confusion matrix: {e}")
    
    # Probability Distribution
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(predicted_probs, bins=30, kde=True)
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Set Threshold ({threshold:.2f})')
        plt.axvline(x=optimal_threshold, color='g', linestyle='--', 
                   label=f'Optimal Threshold ({optimal_threshold:.2f})')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Disruption Probabilities')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            prob_path = os.path.join(output_path, 'disruption_probabilities.png')
            plt.savefig(prob_path)
            visualization_paths['probability_distribution'] = prob_path
        else:
            plt.show()
        plt.close()
    except Exception as e:
        warnings.warn(f"Error generating probability distribution: {e}")
    
    # Group-based analysis if available
    if entity_ids is not None:
        try:
            # Entity-wise accuracy
            plt.figure(figsize=(12, 8))
            
            entity_metrics = {}
            for entity_id in np.unique(entity_ids):
                entity_mask = (entity_ids == entity_id)
                if np.sum(entity_mask) >= 5:  # Require at least 5 data points
                    entity_actual = actual[entity_mask]
                    entity_pred = binary_predictions[entity_mask]
                    entity_metrics[entity_id] = {
                        'accuracy': np.mean(entity_actual == entity_pred),
                        'sample_size': np.sum(entity_mask)
                    }
            
            if entity_metrics:
                # Sort entities by accuracy
                sorted_entities = sorted(entity_metrics.items(), 
                                        key=lambda x: x[1]['accuracy'], 
                                        reverse=True)
                
                entity_ids_sorted = [str(e[0]) for e in sorted_entities]
                accuracies = [e[1]['accuracy'] for e in sorted_entities]
                sample_sizes = [e[1]['sample_size'] for e in sorted_entities]
                
                # Limit to top 20 entities if there are many
                if len(entity_ids_sorted) > 20:
                    entity_ids_sorted = entity_ids_sorted[:20]
                    accuracies = accuracies[:20]
                    sample_sizes = sample_sizes[:20]
                
                y_pos = np.arange(len(entity_ids_sorted))
                
                # Create horizontal bar chart
                plt.barh(y_pos, accuracies)
                plt.yticks(y_pos, entity_ids_sorted)
                plt.xlabel('Prediction Accuracy')
                plt.ylabel('Entity ID')
                plt.title('Disruption Prediction Accuracy by Entity')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                if output_path:
                    entity_path = os.path.join(output_path, 'disruption_entity_accuracy.png')
                    plt.savefig(entity_path)
                    visualization_paths['entity_accuracy'] = entity_path
                else:
                    plt.show()
                plt.close()
        except Exception as e:
            warnings.warn(f"Error generating entity analysis: {e}")
    
    return visualization_paths


def cross_validate_forecast_model(
    model_function,
    data: pd.DataFrame,
    n_splits: int = 5,
    date_col: str = 'date',
    target_col: str = 'demand',
    group_cols: Optional[List[str]] = None,
    gap: int = 0
) -> Dict:
    """
    Perform time-series cross-validation for a forecasting model.
    
    Args:
        model_function: Function that takes training data and returns a prediction function
        data: DataFrame with time series data
        n_splits: Number of cross-validation splits
        date_col: Column name for date
        target_col: Column name for target variable
        group_cols: Optional list of columns to group by (e.g., ['product_id', 'retailer_id'])
        gap: Number of time steps to exclude between train and test sets
        
    Returns:
        Dictionary with cross-validation metrics
    """
    if n_splits < 2:
        warnings.warn("n_splits must be at least 2")
        n_splits = 2
    
    # Ensure data is sorted by date
    data = data.sort_values(by=date_col).reset_index(drop=True)
    
    # Get unique dates
    unique_dates = data[date_col].unique()
    if len(unique_dates) < n_splits * 2:
        warnings.warn("Not enough time periods for requested number of splits")
        n_splits = max(2, len(unique_dates) // 2)
    
    # Calculate split points
    split_indices = []
    train_size = len(unique_dates) // n_splits
    for i in range(1, n_splits):
        split_date = unique_dates[i * train_size]
        split_indices.append(data[data[date_col] >= split_date].index[0])
    
    # Store metrics from each split
    all_metrics = []
    
    # Perform cross-validation
    for i in range(n_splits - 1):
        start_idx = 0 if i == 0 else split_indices[i - 1]
        train_end_idx = split_indices[i]
        
        # Apply gap if specified
        if gap > 0:
            test_start_date = data.loc[train_end_idx, date_col] + pd.Timedelta(days=gap)
            test_start_idx = data[data[date_col] >= test_start_date].index[0] if not data[data[date_col] >= test_start_date].empty else train_end_idx
        else:
            test_start_idx = train_end_idx
        
        test_end_idx = split_indices[i + 1] if i < n_splits - 2 else len(data)
        
        # Get train and test sets
        train_data = data.iloc[start_idx:train_end_idx].copy()
        test_data = data.iloc[test_start_idx:test_end_idx].copy()
        
        # Skip if train or test is empty
        if len(train_data) == 0 or len(test_data) == 0:
            continue
        
        # Train model and get prediction function
        try:
            predict_func = model_function(train_data)
            
            # Make predictions
            predictions = predict_func(test_data)
            
            # Evaluate predictions
            eval_results = evaluate_forecast_model(
                test_data, 
                predictions, 
                date_col=date_col,
                actual_col=target_col,
                forecast_col='forecast',
                group_cols=group_cols
            )
            
            # Store metrics
            split_metrics = {
                'split': i + 1,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'train_start_date': train_data[date_col].min(),
                'train_end_date': train_data[date_col].max(),
                'test_start_date': test_data[date_col].min(),
                'test_end_date': test_data[date_col].max(),
                **eval_results['overall']
            }
            
            all_metrics.append(split_metrics)
            
        except Exception as e:
            warnings.warn(f"Error in cross-validation split {i + 1}: {e}")
    
    # Calculate average metrics
    if not all_metrics:
        return {'error': 'No successful cross-validation splits'}
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if key not in ['split', 'train_size', 'test_size', 'train_start_date', 
                     'train_end_date', 'test_start_date', 'test_end_date']:
            avg_metrics[f'avg_{key}'] = np.mean([m[key] for m in all_metrics])
            avg_metrics[f'std_{key}'] = np.std([m[key] for m in all_metrics])
    
    return {
        'splits': all_metrics,
        'summary': avg_metrics
    }