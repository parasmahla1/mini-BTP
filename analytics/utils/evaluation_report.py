"""
Comprehensive evaluation of prediction accuracy for supply chain analytics.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def evaluate_forecast_accuracy(actual_data, forecast_data, output_dir=None):
    """
    Evaluate demand forecast accuracy.
    
    Args:
        actual_data (pd.DataFrame): DataFrame with actual demand values
        forecast_data (pd.DataFrame): DataFrame with forecasted demand values
        output_dir (str): Optional directory to save visualizations
        
    Returns:
        dict: Dictionary of accuracy metrics
    """
    print("\n" + "="*80)
    print("DEMAND FORECAST ACCURACY EVALUATION".center(80))
    print("="*80)
    
    # Ensure date columns are datetime
    for df in [actual_data, forecast_data]:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
    
    # Identify key columns
    actual_col = 'demand' if 'demand' in actual_data.columns else 'actual_demand'
    forecast_col = 'forecasted_demand' if 'forecasted_demand' in forecast_data.columns else 'predicted_demand'
    
    # Merge datasets on common columns
    merge_cols = list(set(actual_data.columns) & set(forecast_data.columns))
    if not merge_cols:
        print("Error: No common columns found between actual and forecast data")
        return {}
    
    merged_data = pd.merge(
        actual_data, 
        forecast_data,
        on=merge_cols,
        how='inner'
    )
    
    if len(merged_data) == 0:
        print("Error: No matching data points found between actual and forecast data")
        return {}
    
    print(f"Found {len(merged_data)} matching data points for evaluation")
    
    # Calculate accuracy metrics
    y_true = merged_data[actual_col]
    y_pred = merged_data[forecast_col]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    non_zero_mask = (y_true != 0)
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    
    # Calculate weighted MAPE (weighted by demand volume)
    weighted_errors = np.abs(y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]
    weights = y_true[non_zero_mask] / y_true[non_zero_mask].sum()
    wape = np.sum(weighted_errors * weights) * 100
    
    # Calculate forecasting bias (negative = underforecast, positive = overforecast)
    bias = np.mean((y_pred - y_true) / y_true[non_zero_mask]) * 100
    
    # Generate accuracy report
    print("\nForecast Accuracy Metrics:")
    print(f"MAE: {mae:.2f} units")
    print(f"RMSE: {rmse:.2f} units")
    print(f"MAPE: {mape:.2f}%")
    print(f"WAPE: {wape:.2f}%")  
    print(f"R²: {r2:.4f}")
    print(f"Bias: {bias:.2f}%")
    
    # Generate visualization
    if output_dir:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        max_val = max(y_true.max(), y_pred.max())
        min_val = min(y_true.min(), y_pred.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('Forecast Accuracy: Actual vs Predicted Demand')
        plt.xlabel('Actual Demand')
        plt.ylabel('Forecasted Demand')
        plt.grid(True, alpha=0.3)
        
        # Add metrics as text
        plt.figtext(0.15, 0.8, f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%\nR²: {r2:.4f}", 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/forecast_accuracy.png")
        plt.close()
        
        # Time series plot for a sample product-retailer combination
        if 'product_id' in merged_data.columns and 'retailer_id' in merged_data.columns:
            sample_product = merged_data['product_id'].iloc[0]
            sample_retailer = merged_data['retailer_id'].iloc[0]
            
            sample_data = merged_data[
                (merged_data['product_id'] == sample_product) & 
                (merged_data['retailer_id'] == sample_retailer)
            ].sort_values('date')
            
            if len(sample_data) > 0:
                plt.figure(figsize=(12, 6))
                plt.plot(sample_data['date'], sample_data[actual_col], 'b-', label='Actual')
                plt.plot(sample_data['date'], sample_data[forecast_col], 'r--', label='Forecast')
                
                plt.title(f'Demand Forecast Accuracy for Product {sample_product} at Retailer {sample_retailer}')
                plt.xlabel('Date')
                plt.ylabel('Demand')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/forecast_accuracy_timeseries.png")
                plt.close()
    
    # Return metrics dictionary
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'WAPE': wape,
        'R2': r2,
        'Bias': bias,
        'Sample_Count': len(merged_data)
    }


def evaluate_disruption_predictions(actual_data, predicted_data, output_dir=None):
    """
    Evaluate disruption prediction accuracy.
    
    Args:
        actual_data (pd.DataFrame): DataFrame with actual disruption indicators
        predicted_data (pd.DataFrame): DataFrame with predicted disruption probabilities
        output_dir (str): Optional directory to save visualizations
        
    Returns:
        dict: Dictionary of accuracy metrics
    """
    print("\n" + "="*80)
    print("DISRUPTION PREDICTION ACCURACY EVALUATION".center(80))
    print("="*80)
    
    # Identify key columns
    actual_col = None
    for col in ['disruption', 'actual_disruption', 'stockout']:
        if col in actual_data.columns:
            actual_col = col
            break
    
    if actual_col is None:
        print("Error: No disruption indicator column found in actual data")
        return {}
        
    pred_prob_col = None
    for col in ['disruption_probability', 'pred_probability', 'probability']:
        if col in predicted_data.columns:
            pred_prob_col = col
            break
    
    if pred_prob_col is None:
        print("Error: No probability column found in prediction data")
        return {}
    
    # Merge datasets on common columns
    merge_cols = list(set(actual_data.columns) & set(predicted_data.columns))
    
    if not merge_cols:
        print("Error: No common columns found between actual and predicted data")
        return {}
    
    merged_data = pd.merge(
        actual_data, 
        predicted_data,
        on=merge_cols,
        how='inner'
    )
    
    if len(merged_data) == 0:
        print("Error: No matching data points found between actual and predicted data")
        return {}
    
    print(f"Found {len(merged_data)} matching data points for evaluation")
    
    # Convert probabilities to binary predictions using 0.5 threshold
    y_true = merged_data[actual_col].astype(int)
    y_prob = merged_data[pred_prob_col]
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Calculate classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    
    # Calculate accuracy
    accuracy = (y_true == y_pred).mean()
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate other useful metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Generate accuracy report
    print("\nDisruption Prediction Accuracy Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall/Sensitivity: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"False Alarm Rate: {false_alarm_rate:.4f}")
    print(f"Miss Rate: {miss_rate:.4f}")
    
    # Generate visualization
    if output_dir:
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        
        plt.title('Confusion Matrix for Disruption Predictions')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/disruption_confusion_matrix.png")
        plt.close()
        
        # Probability distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=merged_data, x=pred_prob_col, hue=actual_col, 
                    bins=20, element="step", stat="density", common_norm=False)
        
        plt.title('Distribution of Predicted Disruption Probabilities')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.legend(['No Disruption', 'Disruption'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/disruption_probability_dist.png")
        plt.close()
    
    # Return metrics dictionary
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Specificity': specificity,
        'False_Alarm_Rate': false_alarm_rate,
        'Miss_Rate': miss_rate,
        'Sample_Count': len(merged_data)
    }


def evaluate_lead_time_accuracy(actual_data, predicted_data, output_dir=None):
    """
    Evaluate lead time prediction accuracy.
    
    Args:
        actual_data (pd.DataFrame): DataFrame with actual lead times
        predicted_data (pd.DataFrame): DataFrame with predicted lead times
        output_dir (str): Optional directory to save visualizations
        
    Returns:
        dict: Dictionary of accuracy metrics
    """
    print("\n" + "="*80)
    print("LEAD TIME PREDICTION ACCURACY EVALUATION".center(80))
    print("="*80)
    
    # Identify key columns
    actual_col = 'lead_time' if 'lead_time' in actual_data.columns else 'actual_lead_time'
    pred_col = None
    
    for col in ['predicted_lead_time', 'lead_time_pred', 'estimated_lead_time']:
        if col in predicted_data.columns:
            pred_col = col
            break
    
    if pred_col is None:
        print("Error: No lead time prediction column found")
        return {}
    
    # Merge datasets on common columns
    merge_cols = list(set(actual_data.columns) & set(predicted_data.columns))
    if not merge_cols:
        print("Error: No common columns found between actual and predicted data")
        return {}
    
    merged_data = pd.merge(
        actual_data, 
        predicted_data,
        on=merge_cols,
        how='inner'
    )
    
    if len(merged_data) == 0:
        print("Error: No matching data points found between actual and predicted data")
        return {}
    
    print(f"Found {len(merged_data)} matching data points for evaluation")
    
    # Calculate accuracy metrics
    y_true = merged_data[actual_col]
    y_pred = merged_data[pred_col]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE
    non_zero_mask = (y_true != 0)
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    
    # Calculate percentage of predictions within X days of actual
    within_1day = np.mean(np.abs(y_true - y_pred) <= 1)
    within_2days = np.mean(np.abs(y_true - y_pred) <= 2)
    within_3days = np.mean(np.abs(y_true - y_pred) <= 3)
    
    # Generate accuracy report
    print("\nLead Time Prediction Accuracy Metrics:")
    print(f"MAE: {mae:.2f} days")
    print(f"RMSE: {rmse:.2f} days")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")
    print(f"Within 1 day: {within_1day:.2%}")
    print(f"Within 2 days: {within_2days:.2%}")
    print(f"Within 3 days: {within_3days:.2%}")
    
    # Generate visualization
    if output_dir:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        max_val = max(y_true.max(), y_pred.max())
        min_val = min(y_true.min(), y_pred.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('Lead Time Accuracy: Actual vs Predicted')
        plt.xlabel('Actual Lead Time (days)')
        plt.ylabel('Predicted Lead Time (days)')
        plt.grid(True, alpha=0.3)
        
        # Add metrics as text
        plt.figtext(0.15, 0.8, f"MAE: {mae:.2f} days\nRMSE: {rmse:.2f} days\nWithin 2 days: {within_2days:.1%}", 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/lead_time_accuracy.png")
        plt.close()
        
        # Error distribution
        plt.figure(figsize=(10, 6))
        errors = y_pred - y_true
        sns.histplot(errors, kde=True, bins=20)
        
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Lead Time Prediction Error Distribution')
        plt.xlabel('Prediction Error (days)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/lead_time_error_dist.png")
        plt.close()
    
    # Return metrics dictionary
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Within_1day': within_1day,
        'Within_2days': within_2days,
        'Within_3days': within_3days,
        'Sample_Count': len(merged_data)
    }


def evaluate_inventory_policy(actual_data, optimized_policies, output_dir=None):
    """
    Evaluate the effectiveness of inventory policies against actual data.
    
    Args:
        actual_data (pd.DataFrame): DataFrame with actual inventory and demand data
        optimized_policies (pd.DataFrame): DataFrame with optimized inventory policies
        output_dir (str): Optional directory to save visualizations
        
    Returns:
        dict: Dictionary of effectiveness metrics
    """
    print("\n" + "="*80)
    print("INVENTORY POLICY EFFECTIVENESS EVALUATION".center(80))
    print("="*80)
    
    # Check if we have the required columns
    required_cols = {
        'actual': ['product_id', 'retailer_id', 'date', 'demand', 'inventory_level'],
        'policy': ['product_id', 'retailer_id', 'safety_stock', 'reorder_point']
    }
    
    for dataset, req_cols in required_cols.items():
        data = actual_data if dataset == 'actual' else optimized_policies
        missing = [col for col in req_cols if col not in data.columns]
        if missing:
            print(f"Error: Missing columns in {dataset} data: {missing}")
            return {}
    
    # Merge policies with actual data
    merged_data = pd.merge(
        actual_data,
        optimized_policies,
        on=['product_id', 'retailer_id'],
        how='inner'
    )
    
    if len(merged_data) == 0:
        print("Error: No matching data points found between actual data and policies")
        return {}
    
    print(f"Found {len(merged_data)} matching data points for evaluation")
    
    # Evaluate policy effectiveness
    # 1. Calculate stockout rate under policy
    merged_data['stockout'] = (merged_data['inventory_level'] < merged_data['demand']).astype(int)
    merged_data['would_stockout'] = (merged_data['inventory_level'] < merged_data['safety_stock']).astype(int)
    
    # 2. Calculate service levels
    actual_service_level = 1 - (merged_data['stockout'].sum() / len(merged_data))
    policy_service_level = 1 - (merged_data['would_stockout'].sum() / len(merged_data))
    
    # 3. Calculate average inventory levels
    actual_avg_inventory = merged_data['inventory_level'].mean()
    
    # 4. Calculate potential inventory savings
    # Assuming the policy is properly followed, inventory would be:
    # reorder_point + order_quantity - average demand when inventory <= reorder_point
    if 'order_quantity' in optimized_policies.columns:
        avg_order_qty = merged_data['order_quantity'].mean()
        potential_avg_inv = merged_data['reorder_point'].mean() + avg_order_qty/2
        potential_savings = actual_avg_inventory - potential_avg_inv
        potential_savings_pct = (potential_savings / actual_avg_inventory) * 100 if actual_avg_inventory > 0 else 0
    else:
        potential_avg_inv = merged_data['reorder_point'].mean()
        potential_savings = actual_avg_inventory - potential_avg_inv
        potential_savings_pct = (potential_savings / actual_avg_inventory) * 100 if actual_avg_inventory > 0 else 0
    
    # 5. Calculate policy adherence
    below_safety = (merged_data['inventory_level'] < merged_data['safety_stock']).mean() * 100
    below_reorder = (merged_data['inventory_level'] < merged_data['reorder_point']).mean() * 100
    
    # Generate effectiveness report
    print("\nInventory Policy Effectiveness Metrics:")
    print(f"Current Service Level: {actual_service_level:.2%}")
    print(f"Potential Service Level with Policy: {policy_service_level:.2%}")
    print(f"Current Avg Inventory: {actual_avg_inventory:.2f} units")
    print(f"Potential Avg Inventory with Policy: {potential_avg_inv:.2f} units")
    print(f"Potential Inventory Reduction: {potential_savings:.2f} units ({potential_savings_pct:.2f}%)")
    print(f"Inventory Below Safety Stock: {below_safety:.2f}%")
    print(f"Inventory Below Reorder Point: {below_reorder:.2f}%")
    
    # Generate visualization
    if output_dir:
        # Sample a few product-retailer combinations
        samples = merged_data[['product_id', 'retailer_id']].drop_duplicates().sample(
            min(5, len(merged_data[['product_id', 'retailer_id']].drop_duplicates()))
        )
        
        for idx, (product_id, retailer_id) in samples.iterrows():
            sample_data = merged_data[
                (merged_data['product_id'] == product_id) & 
                (merged_data['retailer_id'] == retailer_id)
            ].sort_values('date')
            
            if len(sample_data) < 2:
                continue
                
            plt.figure(figsize=(12, 6))
            
            # Plot inventory level
            plt.plot(sample_data['date'], sample_data['inventory_level'], 
                   'b-', label='Actual Inventory')
            
            # Plot safety stock line
            plt.axhline(y=sample_data['safety_stock'].iloc[0], color='r', 
                      linestyle='-', label='Safety Stock')
            
            # Plot reorder point line
            plt.axhline(y=sample_data['reorder_point'].iloc[0], color='g', 
                      linestyle='--', label='Reorder Point')
            
            # Plot demand
            plt.plot(sample_data['date'], sample_data['demand'], 
                   'k:', label='Demand', alpha=0.6)
            
            plt.title(f'Inventory Policy Analysis for Product {product_id} at {retailer_id}')
            plt.xlabel('Date')
            plt.ylabel('Units')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/inventory_policy_{product_id}_{retailer_id}.png")
            plt.close()
    
    # Return metrics dictionary
    return {
        'Current_Service_Level': actual_service_level,
        'Policy_Service_Level': policy_service_level,
        'Current_Avg_Inventory': actual_avg_inventory,
        'Policy_Avg_Inventory': potential_avg_inv,
        'Potential_Inventory_Reduction_Pct': potential_savings_pct,
        'Inventory_Below_Safety_Stock_Pct': below_safety,
        'Inventory_Below_Reorder_Point_Pct': below_reorder,
        'Sample_Count': len(merged_data)
    }


def run_comprehensive_evaluation(simulation_dir, analytics_dir, output_dir):
    """
    Run comprehensive evaluation of all prediction models.
    
    Args:
        simulation_dir (str): Directory with simulation data (ground truth)
        analytics_dir (str): Directory with analytics predictions
        output_dir (str): Directory to save evaluation results and visualizations
    """
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE MODEL ACCURACY EVALUATION".center(80))
    print("="*80)
    
    results = {}
    
    # 1. Evaluate demand forecasts
    try:
        actual_demand = pd.read_csv(f"{simulation_dir}/demand.csv")
        forecast_demand = pd.read_csv(f"{analytics_dir}/demand_forecast.csv")
        
        results['demand_forecast'] = evaluate_forecast_accuracy(
            actual_demand, forecast_demand, output_dir
        )
    except Exception as e:
        print(f"Error evaluating demand forecast: {e}")
    
    # 2. Evaluate disruption predictions
    try:
        # Create disruption column in inventory data if it doesn't exist
        inventory_data = pd.read_csv(f"{simulation_dir}/inventory.csv")
        if 'disruption' not in inventory_data.columns:
            # Define disruption as inventory level = 0 for products
            if 'item_type' in inventory_data.columns:
                inventory_data['disruption'] = (
                    (inventory_data['inventory_level'] == 0) & 
                    (inventory_data['item_type'] == 'product')
                ).astype(int)
            else:
                inventory_data['disruption'] = (inventory_data['inventory_level'] == 0).astype(int)
            
        disruption_preds = pd.read_csv(f"{analytics_dir}/disruption_predictions.csv")
        
        results['disruption_prediction'] = evaluate_disruption_predictions(
            inventory_data, disruption_preds, output_dir
        )
    except Exception as e:
        print(f"Error evaluating disruption predictions: {e}")
    
    # 3. Evaluate lead time predictions
    try:
        lead_times = pd.read_csv(f"{analytics_dir}/lead_times.csv")
        lead_time_preds = pd.read_csv(f"{analytics_dir}/lead_time_predictions_sample.csv")
        
        results['lead_time_prediction'] = evaluate_lead_time_accuracy(
            lead_times, lead_time_preds, output_dir
        )
    except Exception as e:
        print(f"Error evaluating lead time predictions: {e}")
    
    # 4. Evaluate inventory policies
    try:
        # Combine inventory and demand data for a complete view
        inventory_data = pd.read_csv(f"{simulation_dir}/inventory.csv")
        demand_data = pd.read_csv(f"{simulation_dir}/demand.csv")
        
        # Match retailer IDs with entity_ids
        if 'retailer_id' in demand_data.columns and 'entity_id' in inventory_data.columns:
            # Find inventory records for retailers
            retailer_inventory = inventory_data[inventory_data['entity_id'].isin(demand_data['retailer_id'].unique())]
            
            # Rename columns for consistency
            retailer_inventory = retailer_inventory.rename(columns={'entity_id': 'retailer_id', 'item_id': 'product_id'})
            
            # Merge with demand data
            evaluation_data = pd.merge(
                retailer_inventory,
                demand_data,
                on=['retailer_id', 'product_id', 'date'],
                how='inner'
            )
            
            # Evaluate policies
            inventory_policies = pd.read_csv(f"{analytics_dir}/inventory_policies.csv")
            
            results['inventory_policy'] = evaluate_inventory_policy(
                evaluation_data, inventory_policies, output_dir
            )
    except Exception as e:
        print(f"Error evaluating inventory policies: {e}")
    
    # Save overall results
    try:
        overall_results = pd.DataFrame.from_dict(results, orient='index')
        overall_results.to_csv(f"{output_dir}/evaluation_summary.csv")
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE - Results saved to:".center(80))
        print(f"{output_dir}/evaluation_summary.csv".center(80))
        print("="*80)
    except Exception as e:
        print(f"Error saving overall results: {e}")
    
    return results