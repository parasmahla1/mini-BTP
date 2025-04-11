"""
Visualization utilities for supply chain forecasts.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_demand_forecast(actual_data, forecast_data, product_id, retailer_id):
    """
    Plot demand forecast vs actual values.
    
    Args:
        actual_data (pd.DataFrame): Historical demand data
        forecast_data (pd.DataFrame): Forecasted demand data
        product_id (str): Product ID to plot
        retailer_id (str): Retailer ID to plot
    """
    plt.figure(figsize=(12, 6))
    
    # Filter data for the specific product and retailer
    actual = actual_data[(actual_data['product_id'] == product_id) & 
                         (actual_data['retailer_id'] == retailer_id)]
    forecast = forecast_data[(forecast_data['product_id'] == product_id) & 
                             (forecast_data['retailer_id'] == retailer_id)]
    
    # Sort by date
    actual = actual.sort_values('date')
    forecast = forecast.sort_values('date')
    
    # Plot actual demand
    plt.plot(actual['date'], actual['demand'], label='Actual Demand', marker='o', color='blue')
    
    # Plot forecasted demand
    plt.plot(forecast['date'], forecast['forecasted_demand'], label='Forecasted Demand', 
             linestyle='--', marker='x', color='red')
    
    plt.title(f'Demand Forecast for Product {product_id} at Retailer {retailer_id}')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt

def plot_inventory_optimization(inventory_policies, historical_inventory):
    """
    Plot inventory optimization results.
    
    Args:
        inventory_policies (pd.DataFrame): Calculated inventory policies
        historical_inventory (pd.DataFrame): Historical inventory data
    """
    # Sample a few product-retailer combinations to visualize
    sample_policies = inventory_policies.sample(min(5, len(inventory_policies)))
    
    fig, axs = plt.subplots(len(sample_policies), 1, figsize=(12, 4 * len(sample_policies)))
    
    if len(sample_policies) == 1:
        axs = [axs]  # Make axs iterable if there's only one subplot
    
    for i, (idx, policy) in enumerate(sample_policies.iterrows()):
        product_id = policy['product_id']
        retailer_id = policy['retailer_id']
        
        # Filter historical inventory
        hist_inv = historical_inventory[
            (historical_inventory['item_id'] == product_id) & 
            (historical_inventory['entity_id'] == retailer_id)
        ].sort_values('date')
        
        if len(hist_inv) > 0:
            # Plot historical inventory
            axs[i].plot(hist_inv['date'], hist_inv['inventory_level'], label='Historical Inventory')
            
            # Plot safety stock line
            axs[i].axhline(y=policy['safety_stock'], color='r', linestyle='-', label='Safety Stock')
            
            # Plot reorder point line
            axs[i].axhline(y=policy['reorder_point'], color='g', linestyle='--', label='Reorder Point')
            
            axs[i].set_title(f'Inventory Policy for Product {product_id} at {retailer_id}')
            axs[i].set_xlabel('Date')
            axs[i].set_ylabel('Inventory Level')
            axs[i].legend()
            axs[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

def plot_disruption_heatmap(disruption_predictions):
    """
    Plot heatmap of predicted disruption probabilities.
    
    Args:
        disruption_predictions (pd.DataFrame): Disruption prediction results
    """
    # Convert to wide format for heatmap
    pivot_data = disruption_predictions.pivot_table(
        index='entity_id',
        columns='date',
        values='disruption_probability'
    )
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_data, cmap='YlOrRd', linewidths=0.5)
    plt.title('Supply Chain Disruption Risk Heatmap')
    plt.xlabel('Date')
    plt.ylabel('Entity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt

def plot_lead_time_distribution(lead_times):
    """
    Plot distribution of lead times.
    
    Args:
        lead_times (pd.DataFrame): DataFrame with lead time data
    """
    plt.figure(figsize=(10, 6))
    
    sns.histplot(lead_times['lead_time'], kde=True, bins=20)
    
    plt.title('Distribution of Lead Times')
    plt.xlabel('Lead Time (Days)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def plot_service_levels(service_levels):
    """
    Plot service level metrics by retailer.
    
    Args:
        service_levels (pd.DataFrame): Service level metrics
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate average service level by retailer
    retailer_sl = service_levels.groupby('retailer_id').agg({
        'fill_rate': 'mean',
        'stockout_rate': 'mean'
    }).reset_index()
    
    # Sort for better visualization
    retailer_sl = retailer_sl.sort_values('fill_rate', ascending=False)
    
    # Create bar chart
    x = np.arange(len(retailer_sl))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, retailer_sl['fill_rate'], width, label='Fill Rate', color='green')
    ax.bar(x + width/2, retailer_sl['stockout_rate'], width, label='Stockout Rate', color='red')
    
    ax.set_title('Service Level Metrics by Retailer')
    ax.set_ylabel('Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(retailer_sl['retailer_id'])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return plt