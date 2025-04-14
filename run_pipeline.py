#!/usr/bin/env python3
"""
Unified pipeline for Supply Chain Simulation and Predictive Analytics.
This script runs the entire process from simulation through analytics.
"""

import os
import argparse
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from analytics.utils.evaluation_report import run_comprehensive_evaluation
from analytics.models.demand_forecaster import DemandForecaster

# Import simulation components
from simulation.config import DEFAULT_CONFIG
from simulation.supply_chain_sim import SupplyChainSimulation
from simulation.entities import Supplier, Manufacturer, Warehouse, Retailer

# Import analytics components
from analytics.models.demand_forecaster import DemandForecaster
from analytics.models.inventory_optimizer import InventoryOptimizer
from analytics.models.disruption_predictor import DisruptionPredictor
from analytics.models.lead_time_estimator import LeadTimeEstimator
from analytics.utils.data_processor import calculate_lead_times, calculate_service_levels
from analytics.visualization.forecast_plots import plot_demand_forecast, plot_inventory_optimization, plot_disruption_heatmap


def setup_directories(base_dir='data'):
    """Create necessary directories for data storage."""
    sim_output_dir = os.path.join(base_dir, 'simulation_output')
    analytics_output_dir = os.path.join(base_dir, 'analytics_output')
    
    os.makedirs(sim_output_dir, exist_ok=True)
    os.makedirs(analytics_output_dir, exist_ok=True)
    
    return sim_output_dir, analytics_output_dir




def run_simulation(config, output_dir):
    """Run the supply chain simulation."""
    print("\n" + "="*80)
    print("Starting Supply Chain Simulation".center(80))
    print("="*80)
    
    start_time = time.time()
    
    # Initialize and run simulation
    sim = SupplyChainSimulation(config)
    sim.run_simulation()
    
    # Save simulation data
    transaction_df = sim.get_transaction_data()
    inventory_df = sim.get_inventory_data()
    demand_df = sim.get_demand_data()
    
    transaction_df.to_csv(os.path.join(output_dir, 'transactions.csv'), index=False)
    inventory_df.to_csv(os.path.join(output_dir, 'inventory.csv'), index=False)
    demand_df.to_csv(os.path.join(output_dir, 'demand.csv'), index=False)
    
    elapsed_time = time.time() - start_time
    
    print(f"\nSimulation completed in {elapsed_time:.2f} seconds.")
    print(f"Generated {len(transaction_df)} transactions, {len(inventory_df)} inventory records, and {len(demand_df)} demand records.")
    print(f"Data saved to {output_dir}")
    
    return transaction_df, inventory_df, demand_df

def calculate_lead_times(transaction_data):
    """
    Calculate lead times from transaction data.
    
    Args:
        transaction_data: DataFrame with transaction data
    
    Returns:
        DataFrame with lead time statistics by entity
    """
    lead_times_list = []
    
    try:
        # Check if required columns are present
        required_cols = ['date', 'entity_id', 'transaction_type']
        if not all(col in transaction_data.columns for col in required_cols):
            print("Warning: Required columns missing for lead time calculation")
            return pd.DataFrame(columns=['entity_id', 'lead_time_avg', 'lead_time_std', 'lead_time_p95'])
        
        # Filter for relevant transactions
        orders = transaction_data[transaction_data['transaction_type'] == 'order'].copy()
        shipments = transaction_data[transaction_data['transaction_type'].isin(['ship', 'receive'])].copy()
        
        # If we don't have both orders and shipments, return empty DataFrame
        if orders.empty or shipments.empty:
            return pd.DataFrame(columns=['entity_id', 'lead_time_avg', 'lead_time_std', 'lead_time_p95'])
        
        # Get unique entities
        entities = transaction_data['entity_id'].unique()
        
        for entity_id in entities:
            entity_orders = orders[orders['entity_id'] == entity_id]
            entity_shipments = shipments[shipments['entity_id'] == entity_id]
            
            if entity_orders.empty or entity_shipments.empty:
                continue
            
            # Calculate lead times based on order and shipment dates
            # For simplicity, we're using a statistical approach here
            order_dates = entity_orders['date'].sort_values().reset_index(drop=True)
            shipment_dates = entity_shipments['date'].sort_values().reset_index(drop=True)
            
            # Match closest dates (simplified approach)
            lead_times = []
            for order_date in order_dates:
                if len(shipment_dates) > 0:
                    # Find shipments that happen after the order
                    valid_shipments = shipment_dates[shipment_dates > order_date]
                    if len(valid_shipments) > 0:
                        # Calculate days between order and first subsequent shipment
                        lead_time = (valid_shipments.iloc[0] - order_date).days
                        if 0 < lead_time < 100:  # Reasonable range check
                            lead_times.append(lead_time)
            
            if lead_times:
                lead_times_list.append({
                    'entity_id': entity_id,
                    'lead_time_avg': np.mean(lead_times),
                    'lead_time_std': np.std(lead_times),
                    'lead_time_p95': np.percentile(lead_times, 95),
                    'lead_time_min': min(lead_times),
                    'lead_time_max': max(lead_times)
                })
    
    except Exception as e:
        print(f"Error calculating lead times: {e}")
    
    # Create DataFrame from list
    lead_times_df = pd.DataFrame(lead_times_list)
    
    # If empty, return DataFrame with expected columns
    if lead_times_df.empty:
        lead_times_df = pd.DataFrame(columns=[
            'entity_id', 'lead_time_avg', 'lead_time_std', 
            'lead_time_p95', 'lead_time_min', 'lead_time_max'
        ])
    
    return lead_times_df
def optimize_inventory(demand_data, lead_times, inventory_data):
    """
    Optimize inventory policies based on demand data and lead times.
    
    Args:
        demand_data: DataFrame with historical demand data
        lead_times: DataFrame with lead time statistics
        inventory_data: DataFrame with inventory level data
    
    Returns:
        DataFrame with recommended inventory policies
    """
    policies = []
    
    try:
        # Check if we have sufficient data to compute policies
        if demand_data.empty or lead_times.empty or inventory_data.empty:
            print("Warning: Insufficient data for inventory optimization")
            return pd.DataFrame(columns=['entity_id', 'item_id', 'reorder_point', 'order_quantity', 'safety_stock'])
        
        # Get unique product-entity combinations
        if 'product_id' in demand_data.columns and 'entity_id' in inventory_data.columns:
            products = demand_data['product_id'].unique() if 'product_id' in demand_data.columns else []
            entities = inventory_data['entity_id'].unique()
            
            for entity_id in entities:
                # Get lead time statistics for this entity
                entity_lead_times = lead_times[lead_times['entity_id'] == entity_id]
                
                if entity_lead_times.empty:
                    # Use average lead time if entity-specific data not available
                    avg_lead_time = lead_times['lead_time_avg'].mean() if not lead_times.empty else 7
                    std_lead_time = lead_times['lead_time_std'].mean() if not lead_times.empty else 2
                else:
                    avg_lead_time = entity_lead_times['lead_time_avg'].iloc[0]
                    std_lead_time = entity_lead_times['lead_time_std'].iloc[0]
                
                # Get entity inventory
                entity_inventory = inventory_data[inventory_data['entity_id'] == entity_id]
                
                for product_id in products:
                    # Filter data for this product
                    product_demand = demand_data[demand_data['product_id'] == product_id] if 'product_id' in demand_data.columns else demand_data
                    
                    if product_demand.empty:
                        continue
                    
                    # Calculate demand statistics
                    avg_daily_demand = product_demand['demand'].mean() if 'demand' in product_demand.columns else 10
                    std_daily_demand = product_demand['demand'].std() if 'demand' in product_demand.columns else 5
                    
                    # Calculate policy parameters
                    service_level_z = 1.96  # ~95% service level
                    lead_time_demand = avg_daily_demand * avg_lead_time
                    lead_time_demand_std = np.sqrt(
                        (avg_lead_time * std_daily_demand**2) + 
                        (avg_daily_demand**2 * std_lead_time**2)
                    )
                    
                    safety_stock = service_level_z * lead_time_demand_std
                    reorder_point = lead_time_demand + safety_stock
                    
                    # Economic Order Quantity calculation (simplified)
                    holding_cost_rate = 0.2  # 20% holding cost per year
                    order_cost = 100  # Fixed cost per order
                    annual_demand = avg_daily_demand * 365.25
                    
                    eoq = np.sqrt((2 * order_cost * annual_demand) / holding_cost_rate)
                    
                    policies.append({
                        'entity_id': entity_id,
                        'item_id': product_id,
                        'avg_daily_demand': avg_daily_demand,
                        'std_daily_demand': std_daily_demand,
                        'lead_time_avg': avg_lead_time,
                        'lead_time_std': std_lead_time,
                        'reorder_point': reorder_point,
                        'order_quantity': eoq,
                        'safety_stock': safety_stock,
                        'service_level': 0.95
                    })
        
    except Exception as e:
        print(f"Error in inventory optimization: {e}")
        import traceback
        print(traceback.format_exc())
    
    # Create DataFrame from list
    policies_df = pd.DataFrame(policies)
    
    # If empty, return DataFrame with expected columns
    if policies_df.empty:
        policies_df = pd.DataFrame(columns=[
            'entity_id', 'item_id', 'avg_daily_demand', 'std_daily_demand',
            'lead_time_avg', 'lead_time_std', 'reorder_point', 
            'order_quantity', 'safety_stock', 'service_level'
        ])
    
    return policies_df

def run_analytics(transaction_data, inventory_data, demand_data, output_dir, forecast_days=30, model_type='cnn_lstm'):
    """Run the predictive analytics on simulation data with enhanced models."""
    print("\n" + "="*80)
    print("Running Predictive Analytics".center(80))
    print("="*80)
    
    start_time = time.time()
    
    # Print column names for debugging
    print("\nChecking data structure:")
    print(f"Transaction data shape: {transaction_data.shape}")
    print(f"Inventory data shape: {inventory_data.shape}")
    print(f"Demand data shape: {demand_data.shape}")
    
    # Print first few rows to understand structure
    print("\nTransaction data sample:")
    print(transaction_data.head(2))
    
    # Ensure date columns are datetime
    for df in [transaction_data, inventory_data, demand_data]:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
    
    # Calculate lead times for orders
    print("\nCalculating lead times from transaction data...")
    lead_times = calculate_lead_times(transaction_data)
    lead_times.to_csv(os.path.join(output_dir, 'lead_times.csv'), index=False)
    print(f"Generated {len(lead_times)} lead time records")
    
    # Train demand forecasting model with improved architecture
    print(f"\nTraining demand forecasting model using {model_type}...")
    
    # Initialize with enhanced parameters
    demand_forecaster = DemandForecaster(
        model_type=model_type,
        sequence_length=21,          # Increased history window (3 weeks)
        batch_size=32,               # Standard batch size
        epochs=20,                  # More training epochs
        use_ensemble=False,          # Single model for now
        model_size='large'          # Use larger model with ~1M params
    )
    
    try:
        demand_forecaster.fit(demand_data)
        
        # Generate demand forecast with confidence intervals
        print(f"Generating {forecast_days}-day demand forecast with confidence intervals...")
        forecast_data = demand_forecaster.predict(demand_data, periods_ahead=forecast_days, return_conf_int=True)
        forecast_data.to_csv(os.path.join(output_dir, 'demand_forecast.csv'), index=False)
        
        # Create sample forecast visualization
        if len(forecast_data) > 0:
            print("Creating forecast visualizations...")
            product_id = demand_data['product_id'].iloc[0] if 'product_id' in demand_data.columns else None
            retailer_id = demand_data['retailer_id'].iloc[0] if 'retailer_id' in demand_data.columns else None
            print("Debug: Forecast data columns:", forecast_data.columns.tolist())
            if 'forecasted_demand' not in forecast_data.columns and 'forecast' in forecast_data.columns:
            # Create a copy of the column with the expected name
                forecast_data['forecasted_demand'] = forecast_data['forecast']
            plt_obj = plot_demand_forecast(demand_data, forecast_data, product_id, retailer_id)
            plt_obj.savefig(os.path.join(output_dir, 'demand_forecast_sample.png'))
            plt.close()
    except Exception as e:
        print(f"Error in demand forecasting: {e}")
        import traceback
        print(traceback.format_exc())
    
    # Optimize inventory policies
    print("\nOptimizing inventory policies...")
    try:
        # Now lead_times is defined above
        inventory_policies = optimize_inventory(demand_data, lead_times, inventory_data)
        inventory_policies.to_csv(os.path.join(output_dir, 'inventory_policies.csv'), index=False)
        print(f"Generated {len(inventory_policies)} inventory policies")
    except Exception as e:
        print(f"Error in inventory optimization: {e}")
        import traceback
        print(traceback.format_exc())
    
    # Calculate metrics and other analytics
    elapsed_time = time.time() - start_time
    print(f"\nAnalytics completed in {elapsed_time:.2f} seconds")

def run_evaluation(sim_output_dir, analytics_output_dir):
    """Run evaluation of analytics results against simulation data."""
    print("\n" + "="*80)
    print("Starting Evaluation of Prediction Accuracy".center(80))
    print("="*80)
    
    start_time = time.time()
    
    # Create output directory for evaluation results
    eval_output_dir = os.path.join(analytics_output_dir, 'evaluation')
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Run comprehensive evaluation
    evaluation_results = run_comprehensive_evaluation(
        sim_output_dir, 
        analytics_output_dir, 
        eval_output_dir
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed_time:.2f} seconds.")
    print(f"Results saved to {eval_output_dir}")

def configure_gpu():
    """Configure TensorFlow to use GPU if available."""
    import tensorflow as tf
    
    # Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check for available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"GPU(s) detected: {len(gpus)}")
        try:
            # Configure TensorFlow to use the first GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            
            # Allow memory growth to avoid consuming all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set up mixed precision for faster training
            if tf.__version__ >= "2.4.0":
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("Using mixed precision policy: mixed_float16")
            
            # Verify GPU is being used
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
                c = tf.matmul(a, b)
            print("GPU test successful")
            print(f"Compute device: {c.device}")
            return True
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU available. Using CPU.")
    
    return False

def main():
    is_gpu_enabled = configure_gpu()
    
    parser = argparse.ArgumentParser(description="Run supply chain simulation and predictive analytics pipeline")
    parser.add_argument("--days", type=int, default=365, help="Number of days to simulate")
    parser.add_argument("--forecast", type=int, default=30, help="Number of days to forecast")
    parser.add_argument("--suppliers", type=int, default=5, help="Number of suppliers")
    parser.add_argument("--manufacturers", type=int, default=3, help="Number of manufacturers")
    parser.add_argument("--warehouses", type=int, default=4, help="Number of warehouses")
    parser.add_argument("--retailers", type=int, default=10, help="Number of retailers")
    parser.add_argument("--model", choices=['cnn_lstm', 'xgboost', 'random_forest'], 
                       default='cnn_lstm', 
                       help="Type of ML model for demand forecasting")
    parser.add_argument("--skip-sim", action='store_true', help="Skip simulation and use existing data")
    parser.add_argument("--skip-analytics", action='store_true', help="Skip analytics and use existing results")
    parser.add_argument("--eval-only", action='store_true', help="Only run evaluation on existing data")
    
    args = parser.parse_args()
    
    # Setup directories
    sim_output_dir, analytics_output_dir = setup_directories()
    
    # If only evaluation is requested, skip to evaluation
    if args.eval_only:
        run_evaluation(sim_output_dir, analytics_output_dir)
        return
    
    # Run simulation unless skipped
    if not args.skip_sim:
        # Configure simulation
        config = DEFAULT_CONFIG.copy()
        config['num_suppliers'] = args.suppliers
        config['num_manufacturers'] = args.manufacturers
        config['num_warehouses'] = args.warehouses
        config['num_retailers'] = args.retailers
        
        # Set simulation time range
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = (datetime.now() + pd.Timedelta(days=args.days)).strftime('%Y-%m-%d')
        config['start_date'] = start_date
        config['end_date'] = end_date
        
        # Run simulation
        transaction_df, inventory_df, demand_df = run_simulation(config, sim_output_dir)
    else:
        # Load existing simulation data
        print("\nSkipping simulation, loading existing data...")
        transaction_df = pd.read_csv(os.path.join(sim_output_dir, 'transactions.csv'))
        inventory_df = pd.read_csv(os.path.join(sim_output_dir, 'inventory.csv'))
        demand_df = pd.read_csv(os.path.join(sim_output_dir, 'demand.csv'))
    
    # Run analytics unless skipped
    if not args.skip_analytics:
        run_analytics(
            transaction_df, 
            inventory_df, 
            demand_df, 
            analytics_output_dir,
            forecast_days=args.forecast,
            model_type=args.model
        )
    else:
        print("\nSkipping analytics, using existing results...")
    
    # Run evaluation of predictions against actual data
    run_evaluation(sim_output_dir, analytics_output_dir)
    
    print("\n" + "="*80)
    print("Pipeline Complete".center(80))
    print("="*80)
    print(f"Supply Chain Simulation: {args.days} days")
    print(f"Demand Forecast: {args.forecast} days")
    print(f"Analytics Results: {analytics_output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()