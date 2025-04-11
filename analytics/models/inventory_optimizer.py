"""
Inventory optimization models for supply chain.
"""
import pandas as pd
import numpy as np
from scipy.stats import norm

class InventoryOptimizer:
    """Optimizes inventory levels across the supply chain."""
    
    def __init__(self, service_level=0.95):
        """
        Initialize inventory optimizer.
        
        Args:
            service_level (float): Target service level (0-1)
        """
        self.service_level = service_level
        self.z_score = norm.ppf(service_level)
        self.lead_time_stats = {}
        self.demand_stats = {}
        
    def analyze_lead_times(self, transaction_data):
        """Calculate lead time statistics from transaction data."""
        # Requires transaction data with order and receipt timestamps
        lead_times = transaction_data.copy()
        
        # Group by product and supply node
        for (product_id, source_id, dest_id), group in lead_times.groupby(['product_id', 'entity_id', 'partner_id']):
            # Calculate the statistics
            mean_lead_time = group['lead_time'].mean()
            std_lead_time = group['lead_time'].std()
            
            self.lead_time_stats[(product_id, source_id, dest_id)] = {
                'mean': mean_lead_time,
                'std': std_lead_time
            }
        
        return self.lead_time_stats
    
    def analyze_demand(self, demand_data):
        """Calculate demand statistics from historical data."""
        # Group by product and retailer
        for (product_id, retailer_id), group in demand_data.groupby(['product_id', 'retailer_id']):
            # Calculate daily demand statistics
            daily_mean = group['demand'].mean()
            daily_std = group['demand'].std()
            
            self.demand_stats[(product_id, retailer_id)] = {
                'daily_mean': daily_mean,
                'daily_std': daily_std
            }
        
        return self.demand_stats
    
    def calculate_safety_stock(self, product_id, retailer_id, source_id):
        """Calculate optimal safety stock levels."""
        # Get demand statistics for this product-retailer
        if (product_id, retailer_id) not in self.demand_stats:
            return 0
        
        demand_mean = self.demand_stats[(product_id, retailer_id)]['daily_mean']
        demand_std = self.demand_stats[(product_id, retailer_id)]['daily_std']
        
        # Get lead time statistics for this product-source-retailer
        if (product_id, source_id, retailer_id) not in self.lead_time_stats:
            # Default to 7 days lead time with 2 days std if no data
            lead_time_mean = 7
            lead_time_std = 2
        else:
            lead_time_mean = self.lead_time_stats[(product_id, source_id, retailer_id)]['mean']
            lead_time_std = self.lead_time_stats[(product_id, source_id, retailer_id)]['std']
        
        # Calculate safety stock using the formula:
        # SS = Z * sqrt(L * Var(D) + D^2 * Var(L))
        # Where Z = service level factor, L = lead time, D = daily demand
        
        safety_stock = self.z_score * np.sqrt(
            lead_time_mean * demand_std**2 + demand_mean**2 * lead_time_std**2
        )
        
        return max(0, round(safety_stock))
    
    def calculate_reorder_point(self, product_id, retailer_id, source_id):
        """Calculate reorder points for each product-location combination."""
        # Get demand and lead time statistics
        if (product_id, retailer_id) not in self.demand_stats:
            return 0
        
        demand_mean = self.demand_stats[(product_id, retailer_id)]['daily_mean']
        
        # Get lead time
        if (product_id, source_id, retailer_id) not in self.lead_time_stats:
            lead_time_mean = 7  # Default
        else:
            lead_time_mean = self.lead_time_stats[(product_id, source_id, retailer_id)]['mean']
        
        # Calculate safety stock
        safety_stock = self.calculate_safety_stock(product_id, retailer_id, source_id)
        
        # Reorder point = Expected demand during lead time + Safety stock
        reorder_point = (demand_mean * lead_time_mean) + safety_stock
        
        return max(0, round(reorder_point))
    
    def optimize_inventory_policies(self, transaction_data, demand_data):
        """Generate optimal inventory policies for all nodes."""
        # Analyze lead times and demand
        if 'lead_time' in transaction_data.columns:
            self.analyze_lead_times(transaction_data)
        self.analyze_demand(demand_data)
        
        # Calculate policies for each product-retailer combination
        policies = []
        
        for (product_id, retailer_id), stats in self.demand_stats.items():
            # Find sources for this retailer (simplification: use all warehouses)
            for source_id in transaction_data['entity_id'].unique():
                if 'W' not in source_id:  # Only consider warehouses
                    continue
                    
                safety_stock = self.calculate_safety_stock(product_id, retailer_id, source_id)
                reorder_point = self.calculate_reorder_point(product_id, retailer_id, source_id)
                
                # Calculate order quantity using economic order quantity formula
                # Simplified version without carrying/ordering costs
                daily_demand = stats['daily_mean']
                order_quantity = round(np.sqrt(2 * daily_demand * 100 / 0.2))  # Simplified EOQ
                
                policies.append({
                    'product_id': product_id,
                    'retailer_id': retailer_id,
                    'source_id': source_id,
                    'daily_demand_mean': daily_demand,
                    'daily_demand_std': stats['daily_std'],
                    'safety_stock': safety_stock,
                    'reorder_point': reorder_point,
                    'order_quantity': order_quantity
                })
        
        return pd.DataFrame(policies)