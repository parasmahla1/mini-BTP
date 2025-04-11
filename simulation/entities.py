"""
Supply chain entities including suppliers, manufacturers, warehouses and retailers.
"""
import random
import numpy as np
from datetime import datetime, timedelta


class SupplyChainEntity:
    """Base class for all supply chain entities."""
    
    def __init__(self, entity_id, name, location):
        self.entity_id = entity_id
        self.name = name
        self.location = location  # (lat, long) or region
        self.inventory = {}
        self.order_history = []
        self.lead_time_mean = 2  # days
        self.lead_time_std = 1  # days
        
    def calculate_lead_time(self):
        """Calculate a realistic lead time with some variability."""
        lead_time = max(1, int(np.random.normal(self.lead_time_mean, self.lead_time_std)))
        return lead_time
    
    def log_transaction(self, transaction_type, product_id, quantity, partner_id=None):
        """Record a transaction in the entity's history."""
        timestamp = datetime.now()
        transaction = {
            'timestamp': timestamp,
            'transaction_type': transaction_type,
            'product_id': product_id,
            'quantity': quantity,
            'partner_id': partner_id,
            'entity_id': self.entity_id  # Add entity_id for easier analysis
        }
        self.order_history.append(transaction)
        return transaction


class Supplier(SupplyChainEntity):
    """Raw material supplier."""
    
    def __init__(self, entity_id, name, location, materials):
        super().__init__(entity_id, name, location)
        self.materials = materials
        self.reliability = random.uniform(0.9, 0.99)  # Reliability score
        self.capacity = {material: random.randint(500, 2000) for material in materials}
        
    def supply_material(self, material_id, quantity_requested):
        """Supply requested material if available within capacity."""
        if material_id not in self.materials:
            return 0
            
        # Simulate supply issues based on reliability
        if random.random() > self.reliability:
            quantity_supplied = int(quantity_requested * random.uniform(0.7, 0.95))
        else:
            quantity_supplied = quantity_requested
            
        # Cap at capacity
        quantity_supplied = min(quantity_supplied, self.capacity[material_id])
        
        self.log_transaction('supply', material_id, quantity_supplied)
        return quantity_supplied


class Manufacturer(SupplyChainEntity):
    """Transforms raw materials into products."""
    
    def __init__(self, entity_id, name, location, products):
        super().__init__(entity_id, name, location)
        self.products = products
        self.raw_materials = {}
        self.production_capacity = random.randint(100, 500)  # units per day
        self.efficiency = random.uniform(0.85, 0.95)
        
        # Define material requirements for each product
        self.bom = {}  # bill of materials
        for product in products:
            # Random number of materials needed (1-3)
            num_materials = random.randint(1, 3)
            # Random material IDs (simplified as strings 'm1', 'm2', etc.)
            materials = {f'm{i}': random.randint(1, 5) for i in range(1, num_materials+1)}
            self.bom[product] = materials
    
    def order_materials(self, supplier, material_id, quantity):
        """Order raw materials from supplier."""
        lead_time = self.calculate_lead_time()
        quantity_received = supplier.supply_material(material_id, quantity)
        
        if material_id not in self.raw_materials:
            self.raw_materials[material_id] = 0
        self.raw_materials[material_id] += quantity_received
        
        self.log_transaction('order', material_id, quantity_received, supplier.entity_id)
        return quantity_received
    
    def produce(self, product_id, quantity):
        """Convert raw materials into finished product."""
        if product_id not in self.products:
            return 0
            
        # Check if we have enough materials according to BOM
        can_produce = quantity
        for material, required_qty in self.bom[product_id].items():
            if material not in self.raw_materials:
                return 0
            available_units = self.raw_materials[material] // required_qty
            can_produce = min(can_produce, available_units)
        
        # Cap at production capacity
        can_produce = min(can_produce, self.production_capacity)
        
        # Apply efficiency factor
        actual_production = int(can_produce * self.efficiency)
        
        # Consume raw materials
        for material, required_qty in self.bom[product_id].items():
            self.raw_materials[material] -= required_qty * actual_production
        
        # Update inventory
        if product_id not in self.inventory:
            self.inventory[product_id] = 0
        self.inventory[product_id] += actual_production
        
        self.log_transaction('produce', product_id, actual_production)
        return actual_production
    
    def ship_product(self, product_id, quantity, destination):
        """Ship product to warehouse or retailer."""
        if product_id not in self.inventory or self.inventory[product_id] < quantity:
            actual_quantity = min(quantity, self.inventory.get(product_id, 0))
        else:
            actual_quantity = quantity
            
        if actual_quantity > 0:
            self.inventory[product_id] -= actual_quantity
            self.log_transaction('ship', product_id, actual_quantity, destination.entity_id)
            
        return actual_quantity


class Warehouse(SupplyChainEntity):
    """Stores and distributes products."""
    
    def __init__(self, entity_id, name, location, capacity):
        super().__init__(entity_id, name, location)
        self.total_capacity = capacity
        self.current_utilization = 0
        self.handling_cost = random.uniform(0.5, 2.0)  # cost per unit
        
    def receive_shipment(self, product_id, quantity, source):
        """Receive products from manufacturer."""
        space_available = self.total_capacity - self.current_utilization
        
        if space_available <= 0:
            return 0
            
        quantity_received = min(quantity, space_available)
        
        if product_id not in self.inventory:
            self.inventory[product_id] = 0
        self.inventory[product_id] += quantity_received
        self.current_utilization += quantity_received
        
        self.log_transaction('receive', product_id, quantity_received, source.entity_id)
        return quantity_received
    
    def ship_to_retailer(self, product_id, quantity, retailer):
        """Ship products to a retailer."""
        if product_id not in self.inventory or self.inventory[product_id] < quantity:
            actual_quantity = min(quantity, self.inventory.get(product_id, 0))
        else:
            actual_quantity = quantity
            
        if actual_quantity > 0:
            self.inventory[product_id] -= actual_quantity
            self.current_utilization -= actual_quantity
            self.log_transaction('ship', product_id, actual_quantity, retailer.entity_id)
            
        return actual_quantity


class Retailer(SupplyChainEntity):
    """Sells products to end consumers."""
    
    def __init__(self, entity_id, name, location):
        super().__init__(entity_id, name, location)
        self.sales_history = []
        self.demand_variability = random.uniform(0.1, 0.4)
        self.base_demand = {f'p{i}': random.randint(10, 100) for i in range(1, 6)}
        self.seasonal_factors = self._generate_seasonal_factors()
        
    def _generate_seasonal_factors(self):
        """Generate seasonal demand factors for the year."""
        # Simple seasonality model with peaks around holidays
        base = np.ones(52)  # 52 weeks
        
        # Summer peak (weeks 22-35)
        base[22:36] *= 1.2
        
        # Holiday season peak (weeks 45-52)
        base[45:] *= 1.5
        
        # Random weekly variations
        base *= np.random.normal(1, 0.1, 52)
        
        return base
        
    def forecast_demand(self, product_id, week):
        """Forecast demand for a product in a given week."""
        if product_id not in self.base_demand:
            return 0
            
        base = self.base_demand[product_id]
        seasonal = self.seasonal_factors[min(week, 51)]
        
        # Add some randomness
        forecast = int(base * seasonal * np.random.normal(1, self.demand_variability))
        return max(0, forecast)
    
    def generate_customer_demand(self, product_id, week):
        """Generate actual customer demand for a given week."""
        forecast = self.forecast_demand(product_id, week)
        
        # Actual demand varies from forecast
        actual_demand = int(forecast * np.random.normal(1, self.demand_variability))
        actual_demand = max(0, actual_demand)
        
        return actual_demand
    
    def receive_shipment(self, product_id, quantity, source):
        """Receive products from warehouse or manufacturer."""
        if product_id not in self.inventory:
            self.inventory[product_id] = 0
        self.inventory[product_id] += quantity
        
        self.log_transaction('receive', product_id, quantity, source.entity_id)
        return quantity
    
    def sell_products(self, product_id, week):
        """Sell products to end customers based on demand."""
        demand = self.generate_customer_demand(product_id, week)
        
        if product_id not in self.inventory:
            self.inventory[product_id] = 0
            
        sold = min(demand, self.inventory[product_id])
        self.inventory[product_id] -= sold
        
        # Calculate stockout
        stockout = demand - sold
        stockout_rate = stockout / demand if demand > 0 else 0
        
        sale_record = {
            'timestamp': datetime.now(),
            'product_id': product_id,
            'demand': demand,
            'sold': sold,
            'stockout': stockout,
            'stockout_rate': stockout_rate,
            'retailer_id': self.entity_id
        }
        
        self.sales_history.append(sale_record)
        self.log_transaction('sale', product_id, sold)
        
        return sale_record