#!/usr/bin/env python3
"""
Supply Chain Simulation and Predictive Analytics Pipeline.
Enhanced version with advanced 1D CNN-LSTM models.
"""

import os
import time
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import simulation modules
from simulation.supply_chain_simulator import SupplyChainSimulator
from simulation.config import DEFAULT_CONFIG

# Import analytics modules
from analytics.models.demand_forecaster import DemandForecaster
from analytics.models.disruption_predictor import DisruptionPredictor
from analytics.models.lead_time_predictor import LeadTimePredictor
from analytics.models.inventory_optimizer import InventoryOptimizer
from analytics.utils.evaluation_report import (
    run_comprehensive_evaluation,
    evaluate_forecast_accuracy,
    evaluate_disruption_predictions,
    evaluate_lead_time_accuracy,
    evaluate_inventory_policy
)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Setup output directories."""
    # Create output directories
    sim_output_dir = os.path.join(os.getcwd(), 'output', 'simulation')
    analytics_output_dir = os.path.join(os.getcwd(), 'output', 'analytics')
    
    os.makedirs(sim_output_dir, exist_ok=True)
    os.makedirs(analytics_output_dir, exist_ok=True)
    
    return sim_output_dir, analytics_output_dir

def run_simulation(config, output_dir):
    """Run supply chain simulation."""
    print("\n" + "="*80)
    print("Running Supply Chain Simulation".center(80))
    print("="*80)
    
    start_time = time.time()
    
    # Initialize simulator
    simulator = SupplyChainSimulator(config)
    
    # Run simulation
    simulator.run()
    
    # Get simulation results
    transaction_df = simulator.get_transaction_data()
    inventory_df = simulator.get_inventory_data()
    demand_df = simulator.get_demand_data()
    
    elapsed_time = time.time() - start_time
    print(f"\nSimulation completed in {elapsed_time:.2f} seconds.")
    
    # Save results
    transaction_df.to_csv(os.path.join(output_dir, 'transactions.csv'), index=False)
    inventory_df.to_csv(os.path.join(output_dir, 'inventory.csv'), index=False)
    demand_df.to_csv(os.path.join(output_dir, 'demand.csv'), index=False)
    
    # Additional simulation data
    disruptions_df = simulator.get_disruption_data()
    if disruptions_df is not None:
        disruptions_df.to_csv(os.path.join(output_dir, 'disruptions.csv'), index=False)
    
    lead_times_df = simulator.get_lead_time_data()
    if lead_times_df is not None:
        lead_times_df.to_csv(os.path.join(output_dir, 'lead_times.csv'), index=False)
    
    print(f"Saved simulation data to {output_dir}")
    
    return transaction_df, inventory_df, demand_df

def run_analytics(transaction_data, inventory_data, demand_data, output_dir, 
                 forecast_days=30, model_type='cnn_lstm', external_data=None):
    """
    Run the predictive analytics on simulation data.
    
    Args:
        transaction_data (pd.DataFrame): Transaction data
        inventory_data (pd.DataFrame): Inventory data
        demand_data (pd.DataFrame): Demand data
        output_dir (str): Output directory
        forecast_days (int): Number of days to forecast
        model_type (str): Model type - 'cnn_lstm', 'xgboost' or 'random_forest'
        external_data (pd.DataFrame): Optional external data for enhanced features
    """
    print("\n" + "="*80)
    print("Running Predictive Analytics Pipeline".center(80))
    print("="*80)
    
    start_time = time.time()
    
    # Get unique retailer and product IDs
    retailers = demand_data['retailer_id'].unique()
    products = demand_data['product_id'].unique()
    
    print(f"\nAnalyzing {len(retailers)} retailers and {len(products)} products.")
    
    # Create sample subset for development/debugging
    sample_retailers = retailers[:min(5, len(retailers))]
    sample_products = products[:min(5, len(products))]
    
    # Step 1: Demand Forecasting
    # -------------------------
    print("\n1. Running Demand Forecasting\n" + "-"*50)
    print(f"Using {model_type} model for demand forecasting")
    
    # Initialize and configure demand forecaster
    demand_forecaster = DemandForecaster(model_type=model_type)
    
    # Set advanced model parameters if using CNN-LSTM
    if model_type == 'cnn_lstm':
        # Configure CNN-LSTM settings based on data size
        if len(demand_data) < 5000:  # Small dataset
            demand_forecaster.sequence_length = 10
            demand_forecaster.batch_size = 8
            demand_forecaster.epochs = 50
        elif len(demand_data) < 20000:  # Medium dataset
            demand_forecaster.sequence_length = 14
            demand_forecaster.batch_size = 16
            demand_forecaster.epochs = 100
        else:  # Large dataset
            demand_forecaster.sequence_length = 21
            demand_forecaster.batch_size = 32
            demand_forecaster.epochs = 150
            
        # Enable ensemble if enough data
        demand_forecaster.use_ensemble = len(demand_data) >= 10000
        
        print(f"CNN-LSTM configured with sequence_length={demand_forecaster.sequence_length}, " +
              f"batch_size={demand_forecaster.batch_size}, epochs={demand_forecaster.epochs}")
        if demand_forecaster.use_ensemble:
            print("Using model ensemble for improved accuracy")
    
    # Check for external data
    if external_data is not None:
        print("Using external indicators for enhanced forecasting")
        demand_forecaster.use_external_indicators = True
    
    # Train demand forecasting model
    print("\nTraining demand forecasting model...")
    demand_forecaster.fit(demand_data, external_data)
    
    # Generate forecasts
    forecast_demand = demand_forecaster.predict(
        demand_data, 
        periods_ahead=forecast_days,
        external_data=external_data
    )
    
    # Save forecasts
    forecast_demand.to_csv(os.path.join(output_dir, 'demand_forecast.csv'), index=False)
    print(f"Saved demand forecasts for {len(forecast_demand)} product-retailer combinations")
    
    # Step 2: Disruption Prediction
    # ----------------------------
    print("\n2. Running Disruption Prediction\n" + "-"*50)
    
    # Combine inventory and demand data to create disruption indicators
    if 'stockout' not in inventory_data.columns:
        print("Creating stockout indicators from inventory and demand data")
        # Create a dataframe with combined inventory and demand data
        combined_data = pd.merge(
            inventory_data,
            demand_data,
            left_on=['date', 'entity_id', 'item_id'],
            right_on=['date', 'retailer_id', 'product_id'],
            how='inner'
        )
        
        # Identify stockouts (demand exceeds inventory)
        combined_data['stockout'] = (combined_data['inventory_level'] < combined_data['demand']).astype(int)
        
        # Summarize stockouts by retailer, product, and date
        disruption_data = combined_data.groupby(['retailer_id', 'product_id', 'date']).agg(
            {'stockout': 'max'}
        ).reset_index()
    else:
        # Use existing stockout indicators
        disruption_data = inventory_data[
            ['entity_id', 'item_id', 'date', 'stockout']
        ].rename(columns={'entity_id': 'retailer_id', 'item_id': 'product_id'})
    
    # Train disruption predictor model
    disruption_predictor = DisruptionPredictor()
    disruption_predictor.fit(disruption_data, demand_data)
    
    # Generate predictions
    disruption_preds = disruption_predictor.predict(disruption_data, demand_data)
    
    # Save predictions
    disruption_preds.to_csv(os.path.join(output_dir, 'disruption_predictions.csv'), index=False)
    print(f"Saved disruption predictions for {len(disruption_preds)} product-retailer combinations")
    
    # Step 3: Lead Time Prediction
    # ---------------------------
    print("\n3. Running Lead Time Prediction\n" + "-"*50)
    
    # Extract lead time data from transactions
    if 'lead_time' not in transaction_data.columns:
        print("Calculating lead times from transaction data")
        # Group transactions by order ID and calculate lead time
        order_times = transaction_data.groupby('order_id').agg({
            'date': ['min', 'max'],
            'from_entity_id': 'first',
            'to_entity_id': 'first',
            'item_id': 'first'
        })
        
        # Flatten the MultiIndex columns
        order_times.columns = ['_'.join(col).strip() for col in order_times.columns.values]
        
        # Calculate lead time in days
        order_times['lead_time'] = (
            pd.to_datetime(order_times['date_max']) - 
            pd.to_datetime(order_times['date_min'])
        ).dt.days
        
        # Create lead times dataframe
        lead_times = order_times[[
            'from_entity_id', 'to_entity_id', 'item_id', 'lead_time'
        ]].rename(columns={
            'from_entity_id': 'supplier_id',
            'to_entity_id': 'buyer_id',
            'item_id': 'product_id'
        })
        
        # Add date
        lead_times['date'] = pd.to_datetime(order_times['date_max'])
    else:
        # Use existing lead time data
        lead_times = transaction_data[
            ['from_entity_id', 'to_entity_id', 'item_id', 'date', 'lead_time']
        ].rename(columns={
            'from_entity_id': 'supplier_id',
            'to_entity_id': 'buyer_id',
            'item_id': 'product_id'
        })
    
    # Save lead times
    lead_times.to_csv(os.path.join(output_dir, 'lead_times.csv'), index=False)
    
    # Generate samples for lead time prediction model
    lead_time_predictor = LeadTimePredictor()
    lead_time_predictor.fit(lead_times, transaction_data)
    
    # Generate predictions for sample test cases
    sample_lead_times = lead_time_predictor.create_test_samples(lead_times)
    lead_time_predictions = lead_time_predictor.predict(sample_lead_times)
    
    # Save lead time predictions
    lead_time_predictions.to_csv(os.path.join(output_dir, 'lead_time_predictions_sample.csv'), index=False)
    print(f"Saved lead time predictions for {len(lead_time_predictions)} supplier-buyer-product combinations")
    
    # Step 4: Inventory Optimization
    # ----------------------------
    print("\n4. Running Inventory Optimization\n" + "-"*50)
    
    # Create inventory optimizer
    inventory_optimizer = InventoryOptimizer()
    
    # Generate optimized inventory policies
    inventory_policies = inventory_optimizer.optimize(
        inventory_data, 
        demand_data, 
        forecast_demand,
        lead_times
    )
    
    # Save inventory policies
    inventory_policies.to_csv(os.path.join(output_dir, 'inventory_policies.csv'), index=False)
    print(f"Saved inventory policies for {len(inventory_policies)} product-retailer combinations")
    
    # Step 5: Generate Visualizations
    # -----------------------------
    print("\n5. Generating Visualizations\n" + "-"*50)
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Visualization 1: Demand Forecast vs Actual for a sample product-retailer
    try:
        sample_product = forecast_demand['product_id'].iloc[0]
        sample_retailer = forecast_demand['retailer_id'].iloc[0]
        
        # Get historical data
        historical = demand_data[
            (demand_data['product_id'] == sample_product) & 
            (demand_data['retailer_id'] == sample_retailer)
        ].sort_values('date')
        
        # Get forecast data
        forecast = forecast_demand[
            (forecast_demand['product_id'] == sample_product) & 
            (forecast_demand['retailer_id'] == sample_retailer)
        ].sort_values('date')
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(
            pd.to_datetime(historical['date']), 
            historical['demand'], 
            'b-', 
            label='Historical Demand'
        )
        
        # Plot forecast
        plt.plot(
            pd.to_datetime(forecast['date']),
            forecast['forecasted_demand'],
            'r--',
            label='Forecast'
        )
        
        plt.title(f'Demand Forecast for Product {sample_product} at Retailer {sample_retailer}')
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(viz_dir, 'demand_forecast_sample.png'))
        plt.close()
        
        print("Created demand forecast visualization")
    except Exception as e:
        print(f"Error creating demand forecast visualization: {e}")
    
    # Visualization 2: Disruption probability heatmap
    try:
        # Create a pivot table of disruption probabilities
        pivot_data = disruption_preds.pivot_table(
            index='product_id',
            columns='retailer_id',
            values='disruption_probability',
            aggfunc='mean'
        )
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_data,
            cmap='YlOrRd',
            annot=False,
            fmt='.2f',
            linewidths=0.5
        )
        
        plt.title('Disruption Probability by Product and Retailer')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(viz_dir, 'disruption_heatmap.png'))
        plt.close()
        
        print("Created disruption probability heatmap")
    except Exception as e:
        print(f"Error creating disruption heatmap: {e}")
    
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
    
    # Create visualization directory
    viz_dir = os.path.join(eval_output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
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
        
        # Run evaluations
        results = {}
        
        # 1. Evaluate demand forecasts
        print("\nEvaluating demand forecast accuracy...")
        forecast_results = evaluate_forecast_accuracy(
            actual_demand, forecast_demand, viz_dir
        )
        results['demand_forecast'] = forecast_results
        
        # 2. Evaluate disruption predictions
        print("\nEvaluating disruption prediction accuracy...")
        # Create disruption column in inventory data if it doesn't exist
        if 'disruption' not in inventory_data.columns:
            # Define disruption as inventory level = 0 for products
            if 'item_type' in inventory_data.columns:
                inventory_data['disruption'] = (
                    (inventory_data['inventory_level'] == 0) & 
                    (inventory_data['item_type'] == 'product')
                ).astype(int)
            else:
                inventory_data['disruption'] = (inventory_data['inventory_level'] == 0).astype(int)
            
        disruption_results = evaluate_disruption_predictions(
            inventory_data, disruption_preds, viz_dir
        )
        results['disruption_prediction'] = disruption_results
        
        # 3. Evaluate lead time predictions
        print("\nEvaluating lead time prediction accuracy...")
        lead_time_results = evaluate_lead_time_accuracy(
            lead_times, lead_time_preds, viz_dir
        )
        results['lead_time_prediction'] = lead_time_results
        
        # 4. Evaluate inventory policies
        print("\nEvaluating inventory policy effectiveness...")
        # Combine inventory and demand data for a complete view
        # Match retailer IDs with entity_ids
        if 'retailer_id' in actual_demand.columns and 'entity_id' in inventory_data.columns:
            # Find inventory records for retailers
            retailer_inventory = inventory_data[inventory_data['entity_id'].isin(actual_demand['retailer_id'].unique())]
            
            # Rename columns for consistency
            retailer_inventory = retailer_inventory.rename(columns={'entity_id': 'retailer_id', 'item_id': 'product_id'})
            
            # Merge with demand data
            evaluation_data = pd.merge(
                retailer_inventory,
                actual_demand,
                on=['retailer_id', 'product_id', 'date'],
                how='inner'
            )
            
            # Evaluate policies
            policy_results = evaluate_inventory_policy(
                evaluation_data, inventory_policies, viz_dir
            )
            results['inventory_policy'] = policy_results
        
        # Save overall results
        results_df = pd.DataFrame.from_dict({k: v for k, v in results.items() if isinstance(v, dict)}, 
                                          orient='index')
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
                    if isinstance(value, (int, float)):
                        f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            # Disruption Prediction Accuracy
            f.write("2. DISRUPTION PREDICTION ACCURACY\n")
            f.write("-" * 30 + "\n")
            if 'disruption_prediction' in results:
                for metric, value in disruption_results.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            # Lead Time Prediction Accuracy
            f.write("3. LEAD TIME PREDICTION ACCURACY\n")
            f.write("-" * 30 + "\n")
            if 'lead_time_prediction' in results:
                for metric, value in lead_time_results.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            # Inventory Policy Effectiveness
            f.write("4. INVENTORY POLICY EFFECTIVENESS\n")
            f.write("-" * 30 + "\n")
            if 'inventory_policy' in results:
                for metric, value in policy_results.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")
            
            f.write("=" * 50 + "\n")
            f.write(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        elapsed_time = time.time() - start_time
        print(f"\nEvaluation completed in {elapsed_time:.2f} seconds.")
        print(f"Results saved to {eval_output_dir}")
        print(f"Detailed report: {os.path.join(eval_output_dir, 'evaluation_report.txt')}")
        
        return results
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {}

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Run supply chain simulation and predictive analytics pipeline")
    parser.add_argument("--days", type=int, default=365, help="Number of days to simulate")
    parser.add_argument("--forecast", type=int, default=30, help="Number of days to forecast")
    parser.add_argument("--suppliers", type=int, default=5, help="Number of suppliers")
    parser.add_argument("--manufacturers", type=int, default=3, help="Number of manufacturers")
    parser.add_argument("--warehouses", type=int, default=4, help="Number of warehouses")
    parser.add_argument("--retailers", type=int, default=10, help="Number of retailers")
    parser.add_argument("--model", choices=['cnn_lstm', 'xgboost', 'random_forest'], 
                        default='cnn_lstm', help="Type of ML model for demand forecasting")
    parser.add_argument("--skip-sim", action='store_true', help="Skip simulation and use existing data")
    parser.add_argument("--skip-analytics", action='store_true', help="Skip analytics and use existing results")
    parser.add_argument("--eval-only", action='store_true', help="Only run evaluation on existing data")
    parser.add_argument("--external-data", type=str, default=None, help="Path to external indicators data CSV")
    parser.add_argument("--no-ensemble", action='store_true', help="Disable ensemble models for CNN-LSTM")
    parser.add_argument("--sequence-length", type=int, default=14, help="Sequence length for CNN-LSTM")
    
    args = parser.parse_args()
    
    # Setup directories
    sim_output_dir, analytics_output_dir = setup_directories()
    
    # Load external data if provided
    external_data = None
    if args.external_data:
        try:
            external_data = pd.read_csv(args.external_data)
            print(f"Loaded external data from {args.external_data}: {len(external_data)} records")
        except Exception as e:
            print(f"Error loading external data: {e}")
    
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
            model_type=args.model,
            external_data=external_data
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
    print(f"Model Type: {args.model}")
    print(f"Analytics Results: {analytics_output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()