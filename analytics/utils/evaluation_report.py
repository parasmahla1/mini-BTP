"""
Enhanced evaluation metrics and report generation for supply chain predictive analytics.
Includes specialized metrics for demand forecasting with 1D CNN-LSTM models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import os
import statsmodels.api as sm
from scipy import stats

def evaluate_forecast_accuracy(actual_data, forecast_data, output_dir=None):
    """
    Calculate forecast accuracy metrics with advanced statistical analysis.
    
    Args:
        actual_data (pd.DataFrame): DataFrame with actual demand values
        forecast_data (pd.DataFrame): DataFrame with forecasted demand values
        output_dir (str, optional): Directory to save visualizations
        
    Returns:
        dict: Dictionary of accuracy metrics
    """
    print("Evaluating forecast accuracy...")
    
    # Check if data exists
    if actual_data is None or forecast_data is None:
        print("ERROR: Missing data for evaluation")
        return {'error': 'Missing data'}
    
    # Check if necessary columns exist
    if 'product_id' not in actual_data.columns or 'retailer_id' not in actual_data.columns:
        print("ERROR: Missing product_id or retailer_id columns in actual data")
        return {'error': 'Missing columns in actual data'}
    
    if 'product_id' not in forecast_data.columns or 'retailer_id' not in forecast_data.columns:
        print("ERROR: Missing product_id or retailer_id columns in forecast data")
        return {'error': 'Missing columns in forecast data'}
    
    # Get column names
    actual_col = [col for col in actual_data.columns if 'demand' in col.lower()][0]
    forecast_col = [col for col in forecast_data.columns if 'forecast' in col.lower()][0]
    
    # Convert dates to datetime if they're not already
    if 'date' in actual_data.columns:
        actual_data['date'] = pd.to_datetime(actual_data['date'])
    
    if 'date' in forecast_data.columns:
        forecast_data['date'] = pd.to_datetime(forecast_data['date'])
    
    # Get overlapping product-retailer combinations
    actual_pairs = set(zip(actual_data['product_id'], actual_data['retailer_id']))
    forecast_pairs = set(zip(forecast_data['product_id'], forecast_data['retailer_id']))
    common_pairs = actual_pairs.intersection(forecast_pairs)
    
    if not common_pairs:
        print("ERROR: No common product-retailer pairs between actual and forecast data")
        return {'error': 'No common product-retailer pairs'}
    
    # Merge data on date, product_id, and retailer_id
    merged_data = pd.merge(
        actual_data,
        forecast_data,
        on=['date', 'product_id', 'retailer_id'],
        how='inner'
    )
    
    if len(merged_data) == 0:
        print("ERROR: No matching data points after merge")
        return {'error': 'No matching data points'}
    
    # Calculate accuracy metrics
    y_true = merged_data[actual_col]
    y_pred = merged_data[forecast_col]
    
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Only consider non-zero actual values to avoid division by zero
    non_zero_actual = merged_data[merged_data[actual_col] > 0]
    if len(non_zero_actual) > 0:
        mape = np.mean(np.abs((non_zero_actual[actual_col] - non_zero_actual[forecast_col]) / non_zero_actual[actual_col])) * 100
    else:
        mape = np.nan
    
    # Calculate SMAPE (Symmetric Mean Absolute Percentage Error)
    smape = np.mean(200.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    
    # Calculate WMAPE (Weighted Mean Absolute Percentage Error)
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100 if np.sum(y_true) > 0 else np.nan
    
    # Calculate Median Absolute Error (MdAE)
    mdae = np.median(np.abs(y_true - y_pred))
    
    # Calculate MDA (Mean Directional Accuracy)
    # First, calculate the changes in actual and predicted values
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    
    # Calculate MDA
    if len(y_true_diff) > 0:
        mda = np.mean((np.sign(y_true_diff) == np.sign(y_pred_diff)))
    else:
        mda = np.nan
    
    # Calculate prediction interval coverage
    residuals = y_true - y_pred
    residual_std = np.std(residuals)
    
    # 95% prediction interval using empirical residuals
    z_value = stats.norm.ppf(0.975)  # 95% confidence interval
    lower_bound = y_pred - z_value * residual_std
    upper_bound = y_pred + z_value * residual_std
    
    # Calculate prediction interval coverage
    pic = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    
    # Calculate autocorrelation of residuals (lag-1)
    if len(residuals) > 1:
        try:
            acf = sm.tsa.acf(residuals, nlags=1)[1]
        except:
            acf = np.nan
    else:
        acf = np.nan
    
    # Create results dictionary
    results = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'SMAPE': smape,
        'WMAPE': wmape,
        'MdAE': mdae,
        'MDA': mda,
        'PIC': pic,
        'ACF_lag1': acf,
        'num_data_points': len(merged_data)
    }
    
    # Calculate by-product metrics
    product_metrics = merged_data.groupby('product_id').apply(
        lambda x: pd.Series({
            'MAE': mean_absolute_error(x[actual_col], x[forecast_col]),
            'RMSE': np.sqrt(mean_squared_error(x[actual_col], x[forecast_col])),
            'count': len(x)
        })
    )
    
    # Calculate by-retailer metrics
    retailer_metrics = merged_data.groupby('retailer_id').apply(
        lambda x: pd.Series({
            'MAE': mean_absolute_error(x[actual_col], x[forecast_col]),
            'RMSE': np.sqrt(mean_squared_error(x[actual_col], x[forecast_col])),
            'count': len(x)
        })
    )
    
    # Save metrics by product and retailer
    results['by_product'] = product_metrics.to_dict()
    results['by_retailer'] = retailer_metrics.to_dict()
    
    # Generate visualizations if output directory provided
    if output_dir:
        try:
            # Create scatterplot of actual vs predicted
            plt.figure(figsize=(10, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            
            # Plot the ideal prediction line
            max_value = max(y_true.max(), y_pred.max())
            plt.plot([0, max_value], [0, max_value], 'r--')
            
            plt.title('Actual vs Predicted Demand')
            plt.xlabel('Actual Demand')
            plt.ylabel('Predicted Demand')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
            plt.close()
            
            # Create histogram of residuals
            plt.figure(figsize=(10, 6))
            plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(x=0, color='r', linestyle='--')
            
            plt.title('Residuals Distribution')
            plt.xlabel('Residual (Actual - Predicted)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, 'residuals_distribution.png'))
            plt.close()
            
            # Plot forecast vs actual for a sample product-retailer pair
            if len(common_pairs) > 0:
                # Get a sample pair
                sample_pair = list(common_pairs)[0]
                sample_product_id, sample_retailer_id = sample_pair
                
                # Filter data for this pair
                sample_actual = actual_data[
                    (actual_data['product_id'] == sample_product_id) & 
                    (actual_data['retailer_id'] == sample_retailer_id)
                ].sort_values('date')
                
                sample_forecast = forecast_data[
                    (forecast_data['product_id'] == sample_product_id) & 
                    (forecast_data['retailer_id'] == sample_retailer_id)
                ].sort_values('date')
                
                # Create time series plot
                plt.figure(figsize=(12, 6))
                
                # Plot actual data
                plt.plot(sample_actual['date'], sample_actual[actual_col], 'b-', label='Actual')
                
                # Plot forecasted data
                plt.plot(sample_forecast['date'], sample_forecast[forecast_col], 'r--', label='Forecast')
                
                # Add confidence interval for forecast
                if len(sample_forecast) > 0:
                    plt.fill_between(
                        sample_forecast['date'],
                        sample_forecast[forecast_col] - z_value * residual_std,
                        sample_forecast[forecast_col] + z_value * residual_std,
                        color='r', alpha=0.1, label='95% Prediction Interval'
                    )
                
                plt.title(f'Demand Forecast vs Actual for Product {sample_product_id}, Retailer {sample_retailer_id}')
                plt.xlabel('Date')
                plt.ylabel('Demand')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save figure
                plt.savefig(os.path.join(output_dir, 'time_series_sample.png'))
                plt.close()
                
            # Create heatmap of errors by product and retailer
            if len(merged_data) >= 10:  # Only create heatmap if sufficient data
                try:
                    # Calculate absolute errors
                    merged_data['abs_error'] = np.abs(merged_data[actual_col] - merged_data[forecast_col])
                    
                    # Create pivot table
                    error_pivot = merged_data.pivot_table(
                        values='abs_error',
                        index='product_id',
                        columns='retailer_id',
                        aggfunc='mean'
                    )
                    
                    # Create heatmap
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(error_pivot, cmap='YlOrRd', annot=True, fmt='.1f')
                    
                    plt.title('Mean Absolute Error by Product and Retailer')
                    plt.tight_layout()
                    
                    # Save figure
                    plt.savefig(os.path.join(output_dir, 'error_heatmap.png'))
                    plt.close()
                except Exception as e:
                    print(f"Error creating error heatmap: {e}")
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
    
    # Print summary
    print(f"Forecast Evaluation Summary:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  SMAPE: {smape:.2f}%")
    print(f"  WMAPE: {wmape:.2f}%")
    print(f"  Mean Directional Accuracy: {mda*100:.2f}%")
    print(f"  Prediction Interval Coverage: {pic*100:.2f}%")
    print(f"  Data points: {len(merged_data)}")
    
    return results

def evaluate_forecast_with_confidence(actual_data, forecast_data, alpha=0.05):
    """
    Evaluate forecasts with confidence intervals and advanced metrics.
    
    Args:
        actual_data (pd.DataFrame): DataFrame with actual values
        forecast_data (pd.DataFrame): DataFrame with forecast values
        alpha (float): Significance level for confidence intervals
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    # Merge data
    df = pd.merge(
        actual_data,
        forecast_data,
        on=['date', 'product_id', 'retailer_id'],
        how='inner'
    )
    
    # Get column names
    actual_col = [col for col in actual_data.columns if 'demand' in col.lower()][0]
    forecast_col = [col for col in forecast_data.columns if 'forecast' in col.lower()][0]
    
    # Calculate basic error metrics
    y_true = df[actual_col]
    y_pred = df[forecast_col]
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE
    non_zero_mask = (y_true != 0)
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    
    # Calculate SMAPE (Symmetric MAPE) - more robust
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    
    # Calculate Weighted MAPE
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    
    # Calculate Mean Directional Accuracy (MDA)
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    mda = np.mean((np.sign(y_true_diff) == np.sign(y_pred_diff)))
    
    # Calculate prediction intervals
    residuals = y_true - y_pred
    residual_std = np.std(residuals)
    
    # Normal distribution quantile for confidence interval
    z_value = stats.norm.ppf(1 - alpha/2)
    
    # Lower and upper bounds for prediction interval
    lower_bound = y_pred - z_value * residual_std
    upper_bound = y_pred + z_value * residual_std
    
    # Calculate prediction interval coverage
    pic = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    
    # Calculate autocorrelation of residuals (lag-1)
    acf = sm.tsa.acf(residuals, nlags=1)[1]
    
    # Calculate relative metrics by retailer and product
    by_retailer = df.groupby('retailer_id').apply(
        lambda x: pd.Series({
            'mae': mean_absolute_error(x[actual_col], x[forecast_col]),
            'mape': np.mean(np.abs((x[actual_col][x[actual_col]!=0] - x[forecast_col][x[actual_col]!=0]) / 
                                   x[actual_col][x[actual_col]!=0])) * 100,
            'count': len(x)
        })
    )
    
    by_product = df.groupby('product_id').apply(
        lambda x: pd.Series({
            'mae': mean_absolute_error(x[actual_col], x[forecast_col]),
            'mape': np.mean(np.abs((x[actual_col][x[actual_col]!=0] - x[forecast_col][x[actual_col]!=0]) / 
                                   x[actual_col][x[actual_col]!=0])) * 100,
            'count': len(x)
        })
    )
    
    # Return comprehensive metrics
    return {
        'basic_metrics': {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'SMAPE': smape,
            'WAPE': wape,
            'R2': r2,
            'MDA': mda,
            'PIC': pic,
            'ACF': acf
        },
        'by_retailer': by_retailer.to_dict(),
        'by_product': by_product.to_dict(),
        'prediction_intervals': {
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist()
        }
    }

def evaluate_disruption_predictions(actual_data, prediction_data, output_dir=None):
    """
    Evaluate disruption prediction accuracy.
    
    Args:
        actual_data (pd.DataFrame): DataFrame with actual disruption indicators
        prediction_data (pd.DataFrame): DataFrame with predicted disruption probabilities
        output_dir (str, optional): Directory to save visualizations
        
    Returns:
        dict: Dictionary of accuracy metrics
    """
    print("Evaluating disruption prediction accuracy...")
    
    # Check if data exists
    if actual_data is None or prediction_data is None:
        print("ERROR: Missing data for evaluation")
        return {'error': 'Missing data'}
    
    # Check if necessary columns exist
    if 'disruption' not in actual_data.columns and 'stockout' not in actual_data.columns:
        print("ERROR: Missing disruption/stockout column in actual data")
        return {'error': 'Missing disruption column in actual data'}
    
    if 'disruption_probability' not in prediction_data.columns:
        print("ERROR: Missing disruption_probability column in prediction data")
        return {'error': 'Missing disruption_probability column in prediction data'}
    
    # Rename columns if needed
    if 'entity_id' in actual_data.columns and 'retailer_id' in prediction_data.columns:
        actual_data = actual_data.rename(columns={'entity_id': 'retailer_id'})
    
    if 'item_id' in actual_data.columns and 'product_id' in prediction_data.columns:
        actual_data = actual_data.rename(columns={'item_id': 'product_id'})
    
    # Convert dates to datetime if they're not already
    if 'date' in actual_data.columns:
        actual_data['date'] = pd.to_datetime(actual_data['date'])
    
    if 'date' in prediction_data.columns:
        prediction_data['date'] = pd.to_datetime(prediction_data['date'])
    
    # Determine disruption column
    disruption_col = 'disruption' if 'disruption' in actual_data.columns else 'stockout'
    
    # Merge data on date, product_id, and retailer_id
    merged_data = pd.merge(
        actual_data,
        prediction_data,
        on=['date', 'product_id', 'retailer_id'],
        how='inner'
    )
    
    if len(merged_data) == 0:
        print("ERROR: No matching data points after merge")
        return {'error': 'No matching data points'}
    
    # Convert disruption probability to binary predictions using 0.5 threshold
    merged_data['predicted_disruption'] = (merged_data['disruption_probability'] >= 0.5).astype(int)
    
    # Calculate accuracy metrics
    y_true = merged_data[disruption_col]
    y_pred = merged_data['predicted_disruption']
    y_prob = merged_data['disruption_probability']
    
    # Basic classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate AUC-ROC if there are both positive and negative classes
    if len(np.unique(y_true)) > 1:
        auc_roc = roc_auc_score(y_true, y_prob)
    else:
        auc_roc = np.nan
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Convert confusion matrix to dictionary
    try:
        cm_dict = {
            'true_negative': cm[0, 0],
            'false_positive': cm[0, 1],
            'false_negative': cm[1, 0],
            'true_positive': cm[1, 1]
        }
    except IndexError:
        cm_dict = {'error': 'Could not calculate confusion matrix'}
    
    # Create results dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm_dict,
        'num_data_points': len(merged_data)
    }
    
    # Calculate by-product metrics
    product_metrics = merged_data.groupby('product_id').apply(
        lambda x: pd.Series({
            'accuracy': accuracy_score(x[disruption_col], x['predicted_disruption']),
            'precision': precision_score(x[disruption_col], x['predicted_disruption'], zero_division=0),
            'recall': recall_score(x[disruption_col], x['predicted_disruption'], zero_division=0),
            'count': len(x)
        })
    )
    
    # Calculate by-retailer metrics
    retailer_metrics = merged_data.groupby('retailer_id').apply(
        lambda x: pd.Series({
            'accuracy': accuracy_score(x[disruption_col], x['predicted_disruption']),
            'precision': precision_score(x[disruption_col], x['predicted_disruption'], zero_division=0),
            'recall': recall_score(x[disruption_col], x['predicted_disruption'], zero_division=0),
            'count': len(x)
        })
    )
    
    # Save metrics by product and retailer
    results['by_product'] = product_metrics.to_dict()
    results['by_retailer'] = retailer_metrics.to_dict()
    
    # Generate visualizations if output directory provided
    if output_dir:
        try:
            # Create confusion matrix visualization
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Disruption Prediction Confusion Matrix')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, 'disruption_confusion_matrix.png'))
            plt.close()
            
            # Create probability distribution by actual class
            plt.figure(figsize=(10, 6))
            
            # Plot probability distribution for actual disruptions
            sns.kdeplot(
                merged_data[merged_data[disruption_col] == 1]['disruption_probability'],
                label='Actual Disruptions',
                shade=True,
                color='red'
            )
            
            # Plot probability distribution for non-disruptions
            sns.kdeplot(
                merged_data[merged_data[disruption_col] == 0]['disruption_probability'],
                label='No Disruptions',
                shade=True,
                color='blue'
            )
            
            plt.xlabel('Predicted Disruption Probability')
            plt.ylabel('Density')
            plt.title('Distribution of Predicted Probabilities by Actual Outcome')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, 'disruption_probability_distribution.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error generating disruption visualizations: {e}")
    
    # Print summary
    print(f"Disruption Prediction Evaluation Summary:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  Data points: {len(merged_data)}")
    
    return results

def evaluate_lead_time_accuracy(actual_data, prediction_data, output_dir=None):
    """
    Evaluate lead time prediction accuracy.
    
    Args:
        actual_data (pd.DataFrame): DataFrame with actual lead time values
        prediction_data (pd.DataFrame): DataFrame with predicted lead time values
        output_dir (str, optional): Directory to save visualizations
        
    Returns:
        dict: Dictionary of accuracy metrics
    """
    print("Evaluating lead time prediction accuracy...")
    
    # Check if data exists
    if actual_data is None or prediction_data is None:
        print("ERROR: Missing data for evaluation")
        return {'error': 'Missing data'}
    
    # Check if necessary columns exist
    if 'lead_time' not in actual_data.columns:
        print("ERROR: Missing lead_time column in actual data")
        return {'error': 'Missing lead_time column in actual data'}
    
    if 'predicted_lead_time' not in prediction_data.columns:
        print("ERROR: Missing predicted_lead_time column in prediction data")
        return {'error': 'Missing predicted_lead_time column in prediction data'}
    
    # Ensure consistent column names
    key_columns = ['supplier_id', 'buyer_id', 'product_id']
    for col in key_columns:
        if col not in actual_data.columns:
            alternate_cols = {
                'supplier_id': ['from_entity_id', 'supplier'],
                'buyer_id': ['to_entity_id', 'retailer_id', 'buyer'],
                'product_id': ['item_id', 'product']
            }
            
            for alt_col in alternate_cols[col]:
                if alt_col in actual_data.columns:
                    actual_data = actual_data.rename(columns={alt_col: col})
                    break
        
        if col not in prediction_data.columns:
            alternate_cols = {
                'supplier_id': ['from_entity_id', 'supplier'],
                'buyer_id': ['to_entity_id', 'retailer_id', 'buyer'],
                'product_id': ['item_id', 'product']
            }
            
            for alt_col in alternate_cols[col]:
                if alt_col in prediction_data.columns:
                    prediction_data = prediction_data.rename(columns={alt_col: col})
                    break
    
    # Merge data on supplier_id, buyer_id, and product_id
    merged_data = pd.merge(
        actual_data,
        prediction_data,
        on=['supplier_id', 'buyer_id', 'product_id'],
        how='inner'
    )
    
    if len(merged_data) == 0:
        print("ERROR: No matching data points after merge")
        return {'error': 'No matching data points'}
    
    # Calculate accuracy metrics
    y_true = merged_data['lead_time']
    y_pred = merged_data['predicted_lead_time']
    
    # Basic regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Only consider non-zero actual values to avoid division by zero
    non_zero_actual = merged_data[merged_data['lead_time'] > 0]
    if len(non_zero_actual) > 0:
        mape = np.mean(np.abs((non_zero_actual['lead_time'] - non_zero_actual['predicted_lead_time']) / non_zero_actual['lead_time'])) * 100
    else:
        mape = np.nan
    
    # Create results dictionary
    results = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'num_data_points': len(merged_data)
    }
    
    # Calculate by-supplier metrics
    supplier_metrics = merged_data.groupby('supplier_id').apply(
        lambda x: pd.Series({
            'mae': mean_absolute_error(x['lead_time'], x['predicted_lead_time']),
            'rmse': np.sqrt(mean_squared_error(x['lead_time'], x['predicted_lead_time'])),
            'count': len(x)
        })
    )
    
    # Calculate by-buyer metrics
    buyer_metrics = merged_data.groupby('buyer_id').apply(
        lambda x: pd.Series({
            'mae': mean_absolute_error(x['lead_time'], x['predicted_lead_time']),
            'rmse': np.sqrt(mean_squared_error(x['lead_time'], x['predicted_lead_time'])),
            'count': len(x)
        })
    )
    
    # Save metrics by supplier and buyer
    results['by_supplier'] = supplier_metrics.to_dict()
    results['by_buyer'] = buyer_metrics.to_dict()
    
    # Generate visualizations if output directory provided
    if output_dir:
        try:
            # Create scatterplot of actual vs predicted
            plt.figure(figsize=(10, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            
            # Plot the ideal prediction line
            max_value = max(y_true.max(), y_pred.max())
            plt.plot([0, max_value], [0, max_value], 'r--')
            
            plt.title('Actual vs Predicted Lead Time')
            plt.xlabel('Actual Lead Time (days)')
            plt.ylabel('Predicted Lead Time (days)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, 'lead_time_actual_vs_predicted.png'))
            plt.close()
            
            # Create histogram of errors
            plt.figure(figsize=(10, 6))
            
            # Calculate errors
            errors = y_true - y_pred
            
            plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(x=0, color='r', linestyle='--')
            
            plt.title('Lead Time Prediction Error Distribution')
            plt.xlabel('Error (Actual - Predicted)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, 'lead_time_error_distribution.png'))
            plt.close()
            
            # Create boxplot of errors by supplier
            if len(supplier_metrics) <= 10:  # Only create if there aren't too many suppliers
                plt.figure(figsize=(12, 6))
                
                # Calculate errors by supplier
                merged_data['error'] = merged_data['lead_time'] - merged_data['predicted_lead_time']
                
                # Create boxplot
                sns.boxplot(x='supplier_id', y='error', data=merged_data)
                
                plt.title('Lead Time Prediction Error by Supplier')
                plt.xlabel('Supplier ID')
                plt.ylabel('Error (Actual - Predicted)')
                plt.axhline(y=0, color='r', linestyle='--')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save figure
                plt.savefig(os.path.join(output_dir, 'lead_time_error_by_supplier.png'))
                plt.close()
            
        except Exception as e:
            print(f"Error generating lead time visualizations: {e}")
    
    # Print summary
    print(f"Lead Time Prediction Evaluation Summary:")
    print(f"  MAE: {mae:.2f} days")
    print(f"  RMSE: {rmse:.2f} days")
    print(f"  R²: {r2:.4f}")
    if not np.isnan(mape):
        print(f"  MAPE: {mape:.2f}%")
    print(f"  Data points: {len(merged_data)}")
    
    return results

def evaluate_inventory_policy(actual_data, policy_data, output_dir=None):
    """
    Evaluate inventory policy effectiveness.
    
    Args:
        actual_data (pd.DataFrame): DataFrame with actual inventory and demand data
        policy_data (pd.DataFrame): DataFrame with inventory policy recommendations
        output_dir (str, optional): Directory to save visualizations
        
    Returns:
        dict: Dictionary of effectiveness metrics
    """
    print("Evaluating inventory policy effectiveness...")
    
    # Check if data exists
    if actual_data is None or policy_data is None:
        print("ERROR: Missing data for evaluation")
        return {'error': 'Missing data'}
    
    # Check if necessary columns exist in actual data
    required_cols = ['inventory_level', 'demand']
    missing_cols = [col for col in required_cols if col not in actual_data.columns]
    if missing_cols:
        print(f"ERROR: Missing columns in actual data: {missing_cols}")
        return {'error': f'Missing columns in actual data: {missing_cols}'}
    
    # Check if necessary columns exist in policy data
    policy_required_cols = ['reorder_point', 'order_quantity']
    policy_missing_cols = [col for col in policy_required_cols if col not in policy_data.columns]
    if policy_missing_cols:
        print(f"ERROR: Missing columns in policy data: {policy_missing_cols}")
        return {'error': f'Missing columns in policy data: {policy_missing_cols}'}
    
    # Ensure consistent column names
    key_columns = ['retailer_id', 'product_id']
    for col in key_columns:
        if col not in actual_data.columns:
            alternate_cols = {
                'retailer_id': ['entity_id', 'buyer_id'],
                'product_id': ['item_id']
            }
            
            for alt_col in alternate_cols[col]:
                if alt_col in actual_data.columns:
                    actual_data = actual_data.rename(columns={alt_col: col})
                    break
    
    # Merge data on retailer_id and product_id
    merged_data = pd.merge(
        actual_data,
        policy_data,
        on=['retailer_id', 'product_id'],
        how='inner'
    )
    
    if len(merged_data) == 0:
        print("ERROR: No matching data points after merge")
        return {'error': 'No matching data points'}
    
    # Calculate stockout rate (actual)
    merged_data['stockout'] = (merged_data['inventory_level'] < merged_data['demand']).astype(int)
    stockout_rate = merged_data['stockout'].mean()
    
    # Calculate service level (1 - stockout rate)
    service_level = 1 - stockout_rate
    
    # Calculate average inventory level
    avg_inventory = merged_data['inventory_level'].mean()
    
    # Calculate inventory turnover ratio
    # (Annual demand / Average inventory)
    # Assuming the data covers a specific period, we'll calculate the turnover rate for that period
    total_demand = merged_data['demand'].sum()
    if avg_inventory > 0:
        inventory_turnover = total_demand / avg_inventory
    else:
        inventory_turnover = np.nan
    
    # Calculate days of supply
    # (Average inventory / Average daily demand)
    avg_daily_demand = merged_data['demand'].mean()
    if avg_daily_demand > 0:
        days_of_supply = avg_inventory / avg_daily_demand
    else:
        days_of_supply = np.nan
    
    # Calculate policy effectiveness
    # How well do the recommended reorder points and quantities match the actual requirements?
    # 1. Calculate how many times inventory fell below reorder point
    merged_data['below_reorder_point'] = (merged_data['inventory_level'] <= merged_data['reorder_point']).astype(int)
    pct_below_reorder = merged_data['below_reorder_point'].mean()
    
    # 2. Check if order quantities are sufficient for demand
    merged_data['order_covers_demand'] = (merged_data['order_quantity'] >= merged_data['demand']).astype(int)
    pct_order_sufficient = merged_data['order_covers_demand'].mean()
    
    # Create results dictionary
    results = {
        'stockout_rate': stockout_rate,
        'service_level': service_level,
        'avg_inventory': avg_inventory,
        'inventory_turnover': inventory_turnover,
        'days_of_supply': days_of_supply,
        'pct_below_reorder': pct_below_reorder,
        'pct_order_sufficient': pct_order_sufficient,
        'num_data_points': len(merged_data)
    }
    
    # Calculate by-product metrics
    product_metrics = merged_data.groupby('product_id').apply(
        lambda x: pd.Series({
            'stockout_rate': x['stockout'].mean(),
            'avg_inventory': x['inventory_level'].mean(),
            'count': len(x)
        })
    )
    
    # Calculate by-retailer metrics
    retailer_metrics = merged_data.groupby('retailer_id').apply(
        lambda x: pd.Series({
            'stockout_rate': x['stockout'].mean(),
            'avg_inventory': x['inventory_level'].mean(),
            'count': len(x)
        })
    )
    
    # Save metrics by product and retailer
    results['by_product'] = product_metrics.to_dict()
    results['by_retailer'] = retailer_metrics.to_dict()
    
    # Generate visualizations if output directory provided
    if output_dir:
        try:
            # Create histogram of inventory levels
            plt.figure(figsize=(10, 6))
            
            # Plot histogram
            plt.hist(merged_data['inventory_level'], bins=30, alpha=0.7, edgecolor='black')
            
            # Add vertical line for average inventory
            plt.axvline(x=avg_inventory, color='r', linestyle='--', label=f'Avg: {avg_inventory:.1f}')
            
            plt.title('Distribution of Inventory Levels')
            plt.xlabel('Inventory Level')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, 'inventory_level_distribution.png'))
            plt.close()
            
            # Create scatter plot of inventory level vs demand
            plt.figure(figsize=(10, 6))
            
            # Plot scatter
            plt.scatter(merged_data['demand'], merged_data['inventory_level'], alpha=0.5)
            
            # Add horizontal line for average inventory
            plt.axhline(y=avg_inventory, color='r', linestyle='--', label=f'Avg Inventory: {avg_inventory:.1f}')
            
            # Add vertical line for average demand
            plt.axvline(x=avg_daily_demand, color='g', linestyle='--', label=f'Avg Demand: {avg_daily_demand:.1f}')
            
            plt.title('Inventory Level vs Demand')
            plt.xlabel('Demand')
            plt.ylabel('Inventory Level')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, 'inventory_vs_demand.png'))
            plt.close()
            
            # Create bar chart of service levels by product
            if len(product_metrics) <= 20:  # Only create if there aren't too many products
                plt.figure(figsize=(12, 6))
                
                # Calculate service level by product
                service_level_by_product = 1 - product_metrics['stockout_rate']
                
                # Sort by service level
                service_level_by_product = service_level_by_product.sort_values()
                
                # Create bar chart
                plt.bar(service_level_by_product.index, service_level_by_product.values)
                
                plt.title('Service Level by Product')
                plt.xlabel('Product ID')
                plt.ylabel('Service Level (1 - Stockout Rate)')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save figure
                plt.savefig(os.path.join(output_dir, 'service_level_by_product.png'))
                plt.close()
            
        except Exception as e:
            print(f"Error generating inventory policy visualizations: {e}")
    
    # Print summary
    print(f"Inventory Policy Evaluation Summary:")
    print(f"  Service Level: {service_level:.4f}")
    print(f"  Stockout Rate: {stockout_rate:.4f}")
    print(f"  Avg Inventory: {avg_inventory:.2f}")
    print(f"  Inventory Turnover: {inventory_turnover:.2f}")
    print(f"  Days of Supply: {days_of_supply:.2f}")
    print(f"  Data points: {len(merged_data)}")
    
    return results

def run_comprehensive_evaluation(sim_output_dir, analytics_output_dir, eval_output_dir=None):
    """
    Run comprehensive evaluation of all analytics results.
    
    Args:
        sim_output_dir (str): Directory with simulation data
        analytics_output_dir (str): Directory with analytics results
        eval_output_dir (str, optional): Directory to save evaluation results
        
    Returns:
        dict: Dictionary with all evaluation results
    """
    # Create evaluation output directory if not provided
    if eval_output_dir is None:
        eval_output_dir = os.path.join(analytics_output_dir, 'evaluation')
        os.makedirs(eval_output_dir, exist_ok=True)
    
    # Create visualization directory
    viz_dir = os.path.join(eval_output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    results = {}
    
    try:
        # Load data files
        print("\nLoading data files for evaluation...")
        
        # Simulation data (ground truth)
        actual_demand = pd.read_csv(os.path.join(sim_output_dir, 'demand.csv'))
        inventory_data = pd.read_csv(os.path.join(sim_output_dir, 'inventory.csv'))
        
        try:
            lead_times = pd.read_csv(os.path.join(sim_output_dir, 'lead_times.csv'))
        except:
            lead_times = pd.read_csv(os.path.join(analytics_output_dir, 'lead_times.csv'))
        
        # Analytics results
        forecast_demand = pd.read_csv(os.path.join(analytics_output_dir, 'demand_forecast.csv'))
        disruption_preds = pd.read_csv(os.path.join(analytics_output_dir, 'disruption_predictions.csv'))
        lead_time_preds = pd.read_csv(os.path.join(analytics_output_dir, 'lead_time_predictions_sample.csv'))
        inventory_policies = pd.read_csv(os.path.join(analytics_output_dir, 'inventory_policies.csv'))
        
        # 1. Evaluate demand forecasts
        forecast_results = evaluate_forecast_accuracy(actual_demand, forecast_demand, viz_dir)
        results['demand_forecast'] = forecast_results
        
        # 2. Evaluate disruption predictions
        disruption_results = evaluate_disruption_predictions(inventory_data, disruption_preds, viz_dir)
        results['disruption_prediction'] = disruption_results
        
        # 3. Evaluate lead time predictions
        lead_time_results = evaluate_lead_time_accuracy(lead_times, lead_time_preds, viz_dir)
        results['lead_time_prediction'] = lead_time_results
        
        # 4. Evaluate inventory policies
        # Combine inventory and demand data
        if 'retailer_id' in actual_demand.columns and 'entity_id' in inventory_data.columns:
            inventory_data = inventory_data.rename(columns={'entity_id': 'retailer_id', 'item_id': 'product_id'})
            
        evaluation_data = pd.merge(
            inventory_data,
            actual_demand,
            on=['retailer_id', 'product_id', 'date'],
            how='inner'
        )
        
        policy_results = evaluate_inventory_policy(evaluation_data, inventory_policies, viz_dir)
        results['inventory_policy'] = policy_results
        
        # 5. Run advanced evaluation with confidence intervals
        try:
            confidence_results = evaluate_forecast_with_confidence(actual_demand, forecast_demand)
            results['forecast_with_confidence'] = confidence_results
        except Exception as e:
            print(f"Error in confidence interval evaluation: {e}")
        
        # Save overall results to CSV
        results_df = pd.DataFrame()
        
        # Combine basic metrics from each evaluation
        for key, value in results.items():
            if isinstance(value, dict) and not key == 'forecast_with_confidence':
                # Extract basic metrics that are scalars
                metrics = {k: v for k, v in value.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
                metrics_df = pd.DataFrame(metrics, index=[key])
                results_df = pd.concat([results_df, metrics_df])
        
        # Save to CSV
        results_df.to_csv(os.path.join(eval_output_dir, 'evaluation_summary.csv'))
        
        # Generate comprehensive evaluation report
        with open(os.path.join(eval_output_dir, 'evaluation_report.txt'), 'w') as f:
            f.write("SUPPLY CHAIN PREDICTIVE ANALYTICS EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Demand Forecast Accuracy
            f.write("1. DEMAND FORECAST ACCURACY\n")
            f.write("-" * 30 + "\n")
            if 'demand_forecast' in results:
                for metric, value in forecast_results.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            # Disruption Prediction Accuracy
            f.write("2. DISRUPTION PREDICTION ACCURACY\n")
            f.write("-" * 30 + "\n")
            if 'disruption_prediction' in results:
                for metric, value in disruption_results.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            # Lead Time Prediction Accuracy
            f.write("3. LEAD TIME PREDICTION ACCURACY\n")
            f.write("-" * 30 + "\n")
            if 'lead_time_prediction' in results:
                for metric, value in lead_time_results.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            # Inventory Policy Effectiveness
            f.write("4. INVENTORY POLICY EFFECTIVENESS\n")
            f.write("-" * 30 + "\n")
            if 'inventory_policy' in results:
                for metric, value in policy_results.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            # Advanced Metrics
            f.write("5. ADVANCED EVALUATION METRICS\n")
            f.write("-" * 30 + "\n")
            if 'forecast_with_confidence' in results:
                for metric, value in results['forecast_with_confidence']['basic_metrics'].items():
                    f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            f.write("=" * 50 + "\n")
            f.write(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nComprehensive evaluation complete.")
        print(f"Results saved to {eval_output_dir}")
        
    except Exception as e:
        print(f"Error during comprehensive evaluation: {e}")
    
    return results