"""
Configuration settings for the supply chain simulation.
"""

DEFAULT_CONFIG = {
    # Simulation time range
    'start_date': '2023-01-01',
    'end_date': '2023-12-31',
    
    # Supply chain structure
    'num_suppliers': 5,
    'num_manufacturers': 3,
    'num_warehouses': 4,
    'num_retailers': 10,
    'num_products': 6,
    'num_materials': 10,
    
    # Disruption events
    'enable_disruptions': True,
    'disruption_probability': 0.01,  # probability of disruption per day
    
    # Seasonality
    'seasonal_patterns': {
        'summer_peak': {'start_week': 22, 'end_week': 35, 'factor': 1.3},
        'holiday_peak': {'start_week': 45, 'end_week': 52, 'factor': 1.5}
    },
    
    # Lead time variability
    'lead_time_variability': 0.3,  # coefficient of variation
    
    # Demand variability
    'demand_variability': 0.2,  # coefficient of variation
    
    # Cost parameters
    'holding_cost_rate': 0.02,  # per unit per day
    'backorder_cost_rate': 0.1,  # per unit per day
    'transportation_cost': 0.5,  # per unit per km
    
    # Output settings
    'output_directory': './data/simulation_output',
    'save_format': 'csv'  # csv or json
}