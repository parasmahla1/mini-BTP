"""
Dashboard visualizations for supply chain performance metrics.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_kpi_summary(demand_data, inventory_data, lead_times):
    """
    Create summary of key performance indicators.
    
    Args:
        demand_data (pd.DataFrame): Demand data
        inventory_data (pd.DataFrame): Inventory data
        lead_times (pd.DataFrame): Lead time data
        
    Returns:
        dict: Dictionary of KPI metrics
    """
    # Service level metrics
    service_level = demand_data['fulfilled'].sum() / demand_data['demand'].sum() if demand_data['demand'].sum() > 0 else 0
    stockout_rate = demand_data['stockout'].sum() / demand_data['demand'].sum() if demand_data['demand'].sum() > 0 else 0
    
    # Inventory metrics
    avg_inventory = inventory_data['inventory_level'].mean()
    
    # Supply chain responsiveness
    avg_lead_time = lead_times['lead_time'].mean() if len(lead_times) > 0 else 0
    
    return {
        'Service Level': f"{service_level:.2%}",
        'Stockout Rate': f"{stockout_rate:.2%}",
        'Average Inventory': f"{avg_inventory:.1f} units",
        'Average Lead Time': f"{avg_lead_time:.1f} days"
    }


def plot_kpi_dashboard(demand_data, inventory_data, lead_times):
    """
    Create a dashboard of key performance indicators.
    
    Args:
        demand_data (pd.DataFrame): Demand data
        inventory_data (pd.DataFrame): Inventory data
        lead_times (pd.DataFrame): Lead time data
        
    Returns:
        plt: Matplotlib figure with KPI dashboard
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Supply Chain Performance Dashboard', fontsize=16)
    
    # 1. Service Level Over Time
    if 'date' in demand_data.columns:
        demand_data['date'] = pd.to_datetime(demand_data['date'])
        service_over_time = demand_data.groupby(pd.Grouper(key='date', freq='W')).agg({
            'demand': 'sum',
            'fulfilled': 'sum'
        })
        service_over_time['service_level'] = service_over_time['fulfilled'] / service_over_time['demand']
        
        axs[0, 0].plot(service_over_time.index, service_over_time['service_level'], marker='o')
        axs[0, 0].set_title('Weekly Service Level')
        axs[0, 0].set_ylabel('Service Level')
        axs[0, 0].set_ylim(0, 1)
        axs[0, 0].grid(True, alpha=0.3)
    
    # 2. Inventory Levels
    if 'date' in inventory_data.columns:
        inventory_data['date'] = pd.to_datetime(inventory_data['date'])
        inventory_over_time = inventory_data.groupby([pd.Grouper(key='date', freq='W'), 'entity_type']).agg({
            'inventory_level': 'sum'
        }).reset_index()
        
        for entity_type, group in inventory_over_time.groupby('entity_type'):
            axs[0, 1].plot(group['date'], group['inventory_level'], label=entity_type)
        
        axs[0, 1].set_title('Weekly Inventory Levels by Entity Type')
        axs[0, 1].set_ylabel('Total Inventory')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
    
    # 3. Lead Time Distribution
    if len(lead_times) > 0:
        axs[1, 0].hist(lead_times['lead_time'], bins=10, alpha=0.7)
        axs[1, 0].set_title('Lead Time Distribution')
        axs[1, 0].set_xlabel('Lead Time (Days)')
        axs[1, 0].set_ylabel('Frequency')
        axs[1, 0].grid(True, alpha=0.3)
    
    # 4. Stockout Analysis
    if 'date' in demand_data.columns and 'product_id' in demand_data.columns:
        stockouts_by_product = demand_data.groupby('product_id').agg({
            'demand': 'sum',
            'stockout': 'sum'
        })
        stockouts_by_product['stockout_rate'] = stockouts_by_product['stockout'] / stockouts_by_product['demand']
        stockouts_by_product = stockouts_by_product.sort_values('stockout_rate', ascending=False).head(10)
        
        axs[1, 1].bar(stockouts_by_product.index, stockouts_by_product['stockout_rate'])
        axs[1, 1].set_title('Top 10 Products by Stockout Rate')
        axs[1, 1].set_xlabel('Product ID')
        axs[1, 1].set_ylabel('Stockout Rate')
        axs[1, 1].grid(True, axis='y', alpha=0.3)
        plt.setp(axs[1, 1].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return plt


def plot_network_flow_sankey(transaction_data, entity_map=None):
    """
    Create a Sankey diagram of supply chain flows.
    Note: This requires the plotly library.
    
    Args:
        transaction_data (pd.DataFrame): Transaction data
        entity_map (dict): Optional mapping of entity IDs to readable names
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with Sankey diagram
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is required for Sankey diagrams. Please install with: pip install plotly")
        return None
    
    # Filter to only include 'ship' transactions
    if 'transaction_type' in transaction_data.columns:
        flow_data = transaction_data[transaction_data['transaction_type'] == 'ship'].copy()
    else:
        flow_data = transaction_data.copy()
    
    # Create node labels
    if entity_map is None:
        entity_map = {}
    
    all_entities = list(set(flow_data['entity_id'].unique()) | set(flow_data['partner_id'].unique()))
    nodes = []
    for entity in all_entities:
        label = entity_map.get(entity, entity)
        nodes.append(label)
    
    # Create node mapping
    node_map = {entity: i for i, entity in enumerate(all_entities)}
    
    # Create source-target-value lists
    sources = []
    targets = []
    values = []
    
    for _, row in flow_data.iterrows():
        source = node_map[row['entity_id']]
        target = node_map[row['partner_id']]
        value = row['quantity']
        
        sources.append(source)
        targets.append(target)
        values.append(value)
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])
    
    fig.update_layout(title_text="Supply Chain Network Flow", font_size=10)
    return fig


def plot_entity_network(supply_chain_network, node_types=None):
    """
    Plot the supply chain network structure (requires networkx).
    
    Args:
        supply_chain_network (dict): Network structure from simulation
        node_types (dict): Optional mapping of node IDs to types (for coloring)
        
    Returns:
        plt: Matplotlib figure with network diagram
    """
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX is required for network diagrams. Please install with: pip install networkx")
        return None
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add edges from the network structure
    for mfg_id, warehouses in supply_chain_network.get('manufacturer_to_warehouse', {}).items():
        for wh_id in warehouses:
            G.add_edge(mfg_id, wh_id)
    
    for wh_id, retailers in supply_chain_network.get('warehouse_to_retailer', {}).items():
        for ret_id in retailers:
            G.add_edge(wh_id, ret_id)
    
    for mfg_id, suppliers in supply_chain_network.get('supplier_to_manufacturer', {}).items():
        for sup_id in suppliers:
            G.add_edge(sup_id, mfg_id)
    
    # Set node colors based on type
    node_colors = []
    if node_types is None:
        node_types = {}
        for node in G.nodes:
            if node.startswith('S'):
                node_types[node] = 'supplier'
            elif node.startswith('M'):
                node_types[node] = 'manufacturer'
            elif node.startswith('W'):
                node_types[node] = 'warehouse'
            elif node.startswith('R'):
                node_types[node] = 'retailer'
    
    color_map = {
        'supplier': 'skyblue',
        'manufacturer': 'lightgreen',
        'warehouse': 'orange',
        'retailer': 'pink'
    }
    
    for node in G.nodes:
        node_type = node_types.get(node, 'unknown')
        color = color_map.get(node_type, 'gray')
        node_colors.append(color)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, with_labels=True, node_color=node_colors, 
                    node_size=800, arrows=True, arrowsize=15,
                    font_weight='bold', font_size=10)
    
    # Create legend
    legend_elements = []
    for node_type, color in color_map.items():
        from matplotlib.patches import Patch
        legend_elements.append(Patch(facecolor=color, label=node_type.capitalize()))
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.axis('off')
    plt.title('Supply Chain Network Structure')
    
    return plt