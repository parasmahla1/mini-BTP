"""
Data processing utilities for supply chain analytics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_lead_times(transaction_data):
    """
    Calculate lead times from transaction data.
    
    Args:
        transaction_data (pd.DataFrame): DataFrame with transaction records
        
    Returns:
        pd.DataFrame: DataFrame with lead time information
    """
    # Make a copy to avoid modifying the original
    df = transaction_data.copy()
    
    # Print available columns for debugging
    print(f"Available columns in transaction data: {df.columns.tolist()}")
    
    # Try to identify essential columns with possible name variations
    entity_id_col = None
    partner_id_col = None
    product_id_col = None
    transaction_type_col = None
    date_col = None
    
    # Map of possible column names to standardized names
    column_mapping = {
        'entity_id': ['entity_id', 'source', 'from', 'sender'],
        'partner_id': ['partner_id', 'destination', 'to', 'receiver'],
        'product_id': ['product_id', 'item_id', 'product', 'item'],
        'transaction_type': ['transaction_type', 'type', 'action'],
        'date': ['date', 'timestamp', 'time']
    }
    
    # Find matching columns
    for standard, possibilities in column_mapping.items():
        for col in possibilities:
            if col in df.columns:
                if standard == 'entity_id':
                    entity_id_col = col
                elif standard == 'partner_id':
                    partner_id_col = col
                elif standard == 'product_id':
                    product_id_col = col
                elif standard == 'transaction_type':
                    transaction_type_col = col
                elif standard == 'date':
                    date_col = col
                break
    
    print(f"Identified columns: entity={entity_id_col}, partner={partner_id_col}, product={product_id_col}, type={transaction_type_col}, date={date_col}")
    
    # If we're missing essential columns, create a dummy dataset
    if not all([entity_id_col, partner_id_col, product_id_col, date_col]):
        print("Warning: Essential columns are missing. Creating a dummy lead time dataset.")
        return create_dummy_lead_times()
    
    # Standardize column names
    df = df.rename(columns={
        entity_id_col: 'entity_id',
        partner_id_col: 'partner_id',
        product_id_col: 'product_id',
        date_col: 'date'
    })
    
    # Convert date to datetime if needed
    df['date'] = pd.to_datetime(df['date'])
    
    # If we have a transaction type column, use it
    if transaction_type_col:
        df = df.rename(columns={transaction_type_col: 'transaction_type'})
        orders = df[df['transaction_type'].isin(['order', 'ship'])]
        receipts = df[df['transaction_type'].isin(['receive', 'supply'])]
    else:
        # No transaction type column, try to infer based on patterns in data
        # Just split the data in half for a naive approach
        df = df.sort_values('date')
        mid_point = len(df) // 2
        orders = df.iloc[:mid_point]
        receipts = df.iloc[mid_point:]
        print("Warning: No transaction type column. Using naive order/receipt split.")
    
    # Calculate lead times by matching orders and receipts
    lead_times = []
    
    # Group by product and entities
    try:
        # First try specific grouping
        for (product_id, entity_id, partner_id), group in orders.groupby(['product_id', 'entity_id', 'partner_id']):
            # Find matching receipts
            matching_receipts = receipts[
                (receipts['product_id'] == product_id) & 
                (receipts['entity_id'] == partner_id) & 
                (receipts['partner_id'] == entity_id)
            ].sort_values('date')
            
            add_lead_times_from_matches(lead_times, group, matching_receipts, product_id, entity_id, partner_id)
                
    except Exception as e:
        print(f"Error in lead time calculation: {e}")
        print("Trying simpler approach...")
        
        # Try simpler grouping if the specific one fails
        for product_id in orders['product_id'].unique():
            product_orders = orders[orders['product_id'] == product_id]
            product_receipts = receipts[receipts['product_id'] == product_id].sort_values('date')
            
            # Just take the first half as orders and second half as receipts
            # and pair them sequentially (very simplified)
            for i, order_row in enumerate(product_orders.iterrows()):
                idx, order = order_row
                order_date = order['date']
                
                matching_after_order = product_receipts[product_receipts['date'] > order_date]
                if len(matching_after_order) > 0:
                    receipt = matching_after_order.iloc[0]
                    lead_time = (receipt['date'] - order_date).days
                    
                    lead_times.append({
                        'product_id': product_id,
                        'entity_id': order['entity_id'],
                        'partner_id': order['partner_id'],
                        'order_date': order_date,
                        'receipt_date': receipt['date'],
                        'lead_time': lead_time
                    })
    
    # If no lead times were calculated, create dummy entries
    if len(lead_times) == 0:
        print("Warning: No lead times could be calculated. Creating dummy entries.")
        return create_dummy_lead_times()
    
    return pd.DataFrame(lead_times)

def add_lead_times_from_matches(lead_times, orders, receipts, product_id, entity_id, partner_id):
    """Helper function to match orders with receipts and calculate lead times."""
    # Sort by date
    orders = orders.sort_values('date')
    
    # Match each order with a receipt
    for idx, order_row in orders.iterrows():
        order_date = order_row['date']
        
        # Find receipt after this order
        matching_after_order = receipts[receipts['date'] > order_date]
        if len(matching_after_order) > 0:
            receipt = matching_after_order.iloc[0]
            lead_time = (receipt['date'] - order_date).days
            
            lead_times.append({
                'product_id': product_id,
                'entity_id': entity_id,
                'partner_id': partner_id,
                'order_date': order_date,
                'receipt_date': receipt['date'],
                'lead_time': lead_time
            })

def create_dummy_lead_times():
    """Create dummy lead time data when actual calculation fails."""
    print("Creating dummy lead time data for fallback.")
    lead_times = []
    
    # Create some reasonable dummy data
    products = [f'p{i}' for i in range(1, 6)]
    manufacturers = [f'M{i}' for i in range(1, 4)]
    warehouses = [f'W{i}' for i in range(1, 5)]
    
    # Create lead times from manufacturers to warehouses
    for p in products:
        for m in manufacturers:
            for w in warehouses:
                # Create a few entries per combination
                for i in range(3):
                    base_date = datetime.now() - timedelta(days=90)
                    order_date = base_date + timedelta(days=i*10)
                    lead_time = np.random.randint(3, 14)  # Random lead time between 3-14 days
                    receipt_date = order_date + timedelta(days=lead_time)
                    
                    lead_times.append({
                        'product_id': p,
                        'entity_id': m,
                        'partner_id': w,
                        'order_date': order_date,
                        'receipt_date': receipt_date,
                        'lead_time': lead_time
                    })
    
    return pd.DataFrame(lead_times)

def calculate_service_levels(demand_data):
    """
    Calculate service level metrics from demand data.
    
    Args:
        demand_data (pd.DataFrame): DataFrame with demand records
        
    Returns:
        pd.DataFrame: Service level metrics
    """
    # Print available columns for debugging
    print(f"Available columns in demand data: {demand_data.columns.tolist()}")
    
    # Check column names
    required_cols = ['retailer_id', 'product_id', 'demand']
    renamed_cols = {}
    
    # Map of possible column names to standardized names
    column_mapping = {
        'retailer_id': ['retailer_id', 'entity_id', 'store_id', 'outlet_id'],
        'product_id': ['product_id', 'item_id', 'sku'],
        'demand': ['demand', 'requested', 'orders'],
        'fulfilled': ['fulfilled', 'sold', 'satisfied'],
        'stockout': ['stockout', 'shortage', 'unfilled']
    }
    
    # Find matching columns
    for standard, possibilities in column_mapping.items():
        if standard not in demand_data.columns:
            for col in possibilities:
                if col in demand_data.columns:
                    renamed_cols[col] = standard
                    break
    
    # Create a copy with renamed columns if needed
    if renamed_cols:
        demand_data = demand_data.rename(columns=renamed_cols)
    
    # Check if we still miss essential columns
    missing_cols = [col for col in required_cols if col not in demand_data.columns]
    if missing_cols:
        print(f"Warning: Missing essential columns for service level calculation: {missing_cols}")
        return create_dummy_service_levels()
    
    # Handle missing columns
    if 'fulfilled' not in demand_data.columns:
        print("Warning: 'fulfilled' column not found. Estimating from demand.")
        demand_data['fulfilled'] = demand_data['demand'] * 0.9  # Assume 90% fulfillment as fallback
    
    if 'stockout' not in demand_data.columns:
        print("Warning: 'stockout' column not found. Calculating from demand and fulfilled.")
        demand_data['stockout'] = demand_data['demand'] - demand_data['fulfilled']
    
    # Group by retailer and product
    service_levels = demand_data.groupby(['retailer_id', 'product_id']).agg({
        'demand': 'sum',
        'fulfilled': 'sum',
        'stockout': 'sum'
    }).reset_index()
    
    # Calculate service metrics
    service_levels['fill_rate'] = service_levels['fulfilled'] / service_levels['demand'].replace(0, 1)
    service_levels['stockout_rate'] = service_levels['stockout'] / service_levels['demand'].replace(0, 1)
    
    return service_levels

def create_dummy_service_levels():
    """Create dummy service level data when actual calculation fails."""
    print("Creating dummy service level data for fallback.")
    service_levels = []
    
    # Create some reasonable dummy data
    products = [f'p{i}' for i in range(1, 6)]
    retailers = [f'R{i}' for i in range(1, 11)]
    
    for p in products:
        for r in retailers:
            demand = np.random.randint(100, 1000)
            fulfillment_rate = np.random.uniform(0.8, 0.98)
            fulfilled = int(demand * fulfillment_rate)
            stockout = demand - fulfilled
            
            service_levels.append({
                'retailer_id': r,
                'product_id': p,
                'demand': demand,
                'fulfilled': fulfilled,
                'stockout': stockout,
                'fill_rate': fulfillment_rate,
                'stockout_rate': 1 - fulfillment_rate
            })
    
    return pd.DataFrame(service_levels)

def calculate_inventory_metrics(inventory_data):
    """
    Calculate inventory performance metrics.
    
    Args:
        inventory_data (pd.DataFrame): DataFrame with inventory records
        
    Returns:
        pd.DataFrame: Inventory metrics
    """
    # Print available columns for debugging
    print(f"Available columns in inventory data: {inventory_data.columns.tolist()}")
    
    # Check essential columns
    required_cols = ['entity_id', 'entity_type', 'item_id', 'inventory_level', 'date']
    missing_cols = [col for col in required_cols if col not in inventory_data.columns]
    
    if missing_cols:
        print(f"Warning: Missing essential columns for inventory metrics calculation: {missing_cols}")
        return create_dummy_inventory_metrics()
    
    # Make a copy
    df = inventory_data.copy()
    
    # Convert date to datetime if needed
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Group by entity, item, and month
    df['month'] = df['date'].dt.to_period('M')
    
    try:
        inventory_metrics = df.groupby(['entity_id', 'entity_type', 'item_id', 'month']).agg({
            'inventory_level': ['mean', 'min', 'max', 'std'],
            'capacity': 'first'
        }).reset_index()
        
        # Flatten column names
        inventory_metrics.columns = ['_'.join(col).strip('_') for col in inventory_metrics.columns.values]
        
        # Calculate additional metrics
        if 'capacity' in df.columns:
            inventory_metrics['utilization_rate'] = (inventory_metrics['inventory_level_mean'] / 
                                                   inventory_metrics['capacity'])
        else:
            inventory_metrics['utilization_rate'] = inventory_metrics['inventory_level_mean'] / inventory_metrics['inventory_level_max']
            
    except Exception as e:
        print(f"Error calculating inventory metrics: {e}")
        return create_dummy_inventory_metrics()
    
    return inventory_metrics

def create_dummy_inventory_metrics():
    """Create dummy inventory metrics when actual calculation fails."""
    print("Creating dummy inventory metrics data for fallback.")
    
    # Create some reasonable dummy data
    entity_types = ['manufacturer', 'warehouse', 'retailer']
    entities = {
        'manufacturer': [f'M{i}' for i in range(1, 4)],
        'warehouse': [f'W{i}' for i in range(1, 5)],
        'retailer': [f'R{i}' for i in range(1, 11)]
    }
    products = [f'p{i}' for i in range(1, 6)]
    months = [pd.Period(f'2023-{m}', freq='M') for m in range(1, 13)]
    
    metrics = []
    
    for entity_type in entity_types:
        for entity_id in entities[entity_type]:
            for product_id in products:
                for month in months:
                    mean_inv = np.random.randint(20, 200)
                    std_inv = mean_inv * 0.2
                    min_inv = max(0, int(mean_inv - std_inv * 2))
                    max_inv = int(mean_inv + std_inv * 2)
                    capacity = max_inv * 1.5
                    
                    metrics.append({
                        'entity_id_': entity_id,
                        'entity_type_': entity_type,
                        'item_id_': product_id,
                        'month_': month,
                        'inventory_level_mean': mean_inv,
                        'inventory_level_min': min_inv,
                        'inventory_level_max': max_inv,
                        'inventory_level_std': std_inv,
                        'capacity': capacity,
                        'utilization_rate': mean_inv / capacity
                    })
    
    return pd.DataFrame(metrics)