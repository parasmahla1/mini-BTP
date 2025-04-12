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


def run_analytics(transaction_data, inventory_data, demand_data, output_dir, forecast_days=30, model_type='cnn_lstm'):
    """Run the predictive analytics on simulation data."""
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
    
    # Train demand forecasting model
    print(f"\nTraining demand forecasting model using {model_type}...")
    
    # Initialize with parameters that the DemandForecaster class accepts
    demand_forecaster = DemandForecaster(
        model_type=model_type,       # Model type is supported now
        sequence_length=14,          # Use 2 weeks of history
        batch_size=32,               
        epochs=100,                  # Max training epochs
        use_ensemble=False           # Set to True if you want ensemble models
    )
    
    try:
        demand_forecaster.fit(demand_data)
        
        # Generate demand forecast
        print(f"Generating {forecast_days}-day demand forecast...")
        forecast_data = demand_forecaster.predict(demand_data, periods_ahead=forecast_days)
        forecast_data.to_csv(os.path.join(output_dir, 'demand_forecast.csv'), index=False)
        
        # Create sample forecast visualization
        if len(forecast_data) > 0:
            print("Creating forecast visualizations...")
            product_id = demand_data['product_id'].iloc[0] if 'product_id' in demand_data.columns else None
            retailer_id = demand_data['retailer_id'].iloc[0] if 'retailer_id' in demand_data.columns else None
            
            plt_obj = plot_demand_forecast(demand_data, forecast_data, product_id, retailer_id)
            plt_obj.savefig(os.path.join(output_dir, 'demand_forecast_sample.png'))
            plt.close()
    except Exception as e:
        print(f"Error in demand forecasting: {e}")
    
    
    # Optimize inventory
    print("\nOptimizing inventory policies...")
    inventory_optimizer = InventoryOptimizer(service_level=0.95)
    
    try:
        # Add lead time to transaction data with a safe merge
        if len(lead_times) > 0:
            # First check if we can merge directly
            common_cols = set(transaction_data.columns) & set(['product_id', 'entity_id', 'partner_id'])
            
            if len(common_cols) == 3:
                transaction_with_lead = transaction_data.merge(
                    lead_times[['product_id', 'entity_id', 'partner_id', 'lead_time']], 
                    on=['product_id', 'entity_id', 'partner_id'],
                    how='left'
                )
            else:
                # If columns don't match, just append lead time data
                print("Warning: Cannot merge lead times with transactions. Using lead time data directly.")
                transaction_with_lead = lead_times
                
            transaction_with_lead['lead_time'] = transaction_with_lead['lead_time'].fillna(7)  # Default lead time
        else:
            transaction_with_lead = transaction_data.copy()
            transaction_with_lead['lead_time'] = 7  # Default lead time
        
        inventory_policies = inventory_optimizer.optimize_inventory_policies(
            transaction_with_lead, demand_data
        )
        inventory_policies.to_csv(os.path.join(output_dir, 'inventory_policies.csv'), index=False)
    except Exception as e:
        print(f"Error in inventory optimization: {e}")
    
    # Predict disruptions
    print("\nPredicting supply chain disruptions...")
    disruption_predictor = DisruptionPredictor()
    
    try:
        disruption_predictor.fit(inventory_data)
        disruptions = disruption_predictor.predict_disruptions(inventory_data)
        disruptions.to_csv(os.path.join(output_dir, 'disruption_predictions.csv'), index=False)
        
        # Create disruption visualization
        plt_obj = plot_disruption_heatmap(disruptions)
        plt_obj.savefig(os.path.join(output_dir, 'disruption_heatmap.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate disruption predictions due to: {e}")
    
    # Predict lead times
    print("\nTraining lead time prediction model...")
    lead_time_estimator = LeadTimeEstimator()
    
    try:
        if 'lead_time' in transaction_with_lead.columns:
            lead_time_estimator.fit(transaction_with_lead)
            lead_time_predictions = lead_time_estimator.predict(transaction_with_lead)
            
            # Save sample predictions
            sample_predictions = transaction_with_lead.iloc[:20].copy()
            sample_predictions['predicted_lead_time'] = lead_time_predictions[:20]
            sample_predictions.to_csv(os.path.join(output_dir, 'lead_time_predictions_sample.csv'), index=False)
    except Exception as e:
        print(f"Warning: Could not generate lead time predictions due to: {e}")
    
    # Calculate service levels
    try:
        service_levels = calculate_service_levels(demand_data)
        service_levels.to_csv(os.path.join(output_dir, 'service_levels.csv'), index=False)
    except Exception as e:
        print(f"Error calculating service levels: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"\nAnalytics completed in {elapsed_time:.2f} seconds.")
    print(f"Results saved to {output_dir}")


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
    
def main():
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