"""
Feature engineering utilities for supply chain analytics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_time_features(df, date_column='date'):
    """
    Create time-based features from date column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_column (str): Name of date column
        
    Returns:
        pd.DataFrame: DataFrame with added time features
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Ensure date column is datetime
    if date_column in result.columns:
        result[date_column] = pd.to_datetime(result[date_column])
        
        # Extract date components
        result['day_of_week'] = result[date_column].dt.dayofweek
        result['day_of_month'] = result[date_column].dt.day
        result['month'] = result[date_column].dt.month
        result['quarter'] = result[date_column].dt.quarter
        result['year'] = result[date_column].dt.year
        result['week_of_year'] = result[date_column].dt.isocalendar().week
        
        # Cyclical encoding of time features
        result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        
        # Flags for weekends, holidays, etc.
        result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
        
        # Simple holiday flag (can be expanded)
        result['is_end_of_month'] = (result[date_column].dt.is_month_end).astype(int)
        result['is_end_of_quarter'] = (result[date_column].dt.is_quarter_end).astype(int)
    
    return result


def create_lag_features(df, group_columns, target_column, lag_periods):
    """
    Create lagged features for time series data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_columns (list): Columns to group by
        target_column (str): Target column to create lags for
        lag_periods (list): List of lag periods to create
        
    Returns:
        pd.DataFrame: DataFrame with added lag features
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Ensure data is sorted by date within each group
    if 'date' in result.columns:
        result = result.sort_values(by=group_columns + ['date'])
    
    # Create lag features for each period
    for lag in lag_periods:
        result[f'{target_column}_lag_{lag}'] = result.groupby(group_columns)[target_column].shift(lag)
    
    return result


def create_rolling_features(df, group_columns, target_column, windows, functions=None):
    """
    Create rolling window features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_columns (list): Columns to group by
        target_column (str): Target column to create rolling features for
        windows (list): List of window sizes
        functions (list): List of functions to apply (default: mean, std, min, max)
        
    Returns:
        pd.DataFrame: DataFrame with added rolling features
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Default functions
    if functions is None:
        functions = ['mean', 'std', 'min', 'max']
    
    # Ensure data is sorted by date within each group
    if 'date' in result.columns:
        result = result.sort_values(by=group_columns + ['date'])
    
    # Create rolling features for each window and function
    for window in windows:
        for func in functions:
            col_name = f'{target_column}_rolling_{window}_{func}'
            
            if func == 'mean':
                result[col_name] = result.groupby(group_columns)[target_column].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean())
            elif func == 'std':
                result[col_name] = result.groupby(group_columns)[target_column].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std())
            elif func == 'min':
                result[col_name] = result.groupby(group_columns)[target_column].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min())
            elif func == 'max':
                result[col_name] = result.groupby(group_columns)[target_column].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max())
    
    return result


def create_supply_chain_features(df, entity_column, timestamp_column, transaction_type_column=None):
    """
    Create specialized features for supply chain analysis.
    
    Args:
        df (pd.DataFrame): Input DataFrame with transaction data
        entity_column (str): Column with entity identifiers
        timestamp_column (str): Column with timestamps
        transaction_type_column (str): Column with transaction types
        
    Returns:
        pd.DataFrame: DataFrame with added supply chain features
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Ensure timestamp column is datetime
    result[timestamp_column] = pd.to_datetime(result[timestamp_column])
    
    # Calculate transaction frequency by entity
    result['days_since_last_transaction'] = result.groupby(entity_column)[timestamp_column].diff().dt.total_seconds() / (24 * 3600)
    
    # Calculate transaction volume features
    if 'quantity' in result.columns:
        # Daily volume by entity
        result['daily_volume'] = result.groupby([entity_column, pd.Grouper(key=timestamp_column, freq='D')])['quantity'].transform('sum')
        
        # Weekly moving average
        result['weekly_avg_volume'] = result.groupby(entity_column)['quantity'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean())
    
    # Transaction type patterns
    if transaction_type_column is not None:
        # Count of each transaction type in last 30 days
        for tx_type in result[transaction_type_column].unique():
            result[f'{tx_type}_count_30d'] = result[result[transaction_type_column] == tx_type].groupby(entity_column).rolling(
                window='30D', on=timestamp_column).size().reset_index(level=0, drop=True)
            result[f'{tx_type}_count_30d'] = result[f'{tx_type}_count_30d'].fillna(0)
    
    # Fill missing values
    result = result.fillna(0)
    
    return result