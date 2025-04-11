"""
Supply chain simulation engine.
"""
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from simulation.entities import Supplier, Manufacturer, Warehouse, Retailer


class SupplyChainSimulation:
    """Main simulation engine for the supply chain."""
    
    def __init__(self, config):
        self.config = config
        self.current_date = datetime.strptime(config['start_date'], '%Y-%m-%d')
        self.end_date = datetime.strptime(config['end_date'], '%Y-%m-%d')
        self.timestep = timedelta(days=1)
        
        # Initialize entities
        self.suppliers = self._create_suppliers()
        self.manufacturers = self._create_manufacturers()
        self.warehouses = self._create_warehouses()
        self.retailers = self._create_retailers()
        
        # Link entities in the supply chain
        self._setup_supply_chain_network()
        
        # Setup data collection
        self.transaction_log = []
        self.inventory_log = []
        self.demand_log = []
        
    def _create_suppliers(self):
        """Create supplier entities."""
        suppliers = []
        materials = [f'm{i}' for i in range(1, self.config['num_materials'] + 1)]
        
        for i in range(self.config['num_suppliers']):
            # Each supplier supplies a random subset of materials
            supplier_materials = random.sample(
                materials, 
                k=random.randint(1, len(materials))
            )
            supplier = Supplier(
                entity_id=f"S{i+1}",
                name=f"Supplier {i+1}",
                location=(
                    random.uniform(-90, 90),  # latitude
                    random.uniform(-180, 180)  # longitude
                ),
                materials=supplier_materials
            )
            suppliers.append(supplier)
        
        return suppliers
    
    def _create_manufacturers(self):
        """Create manufacturer entities."""
        manufacturers = []
        products = [f'p{i}' for i in range(1, self.config['num_products'] + 1)]
        
        for i in range(self.config['num_manufacturers']):
            # Each manufacturer produces a subset of products
            manufacturer_products = random.sample(
                products, 
                k=random.randint(1, len(products))
            )
            manufacturer = Manufacturer(
                entity_id=f"M{i+1}",
                name=f"Manufacturer {i+1}",
                location=(
                    random.uniform(-90, 90),
                    random.uniform(-180, 180)
                ),
                products=manufacturer_products
            )
            manufacturers.append(manufacturer)
        
        return manufacturers
    
    def _create_warehouses(self):
        """Create warehouse entities."""
        warehouses = []
        
        for i in range(self.config['num_warehouses']):
            capacity = random.randint(1000, 5000)
            warehouse = Warehouse(
                entity_id=f"W{i+1}",
                name=f"Warehouse {i+1}",
                location=(
                    random.uniform(-90, 90),
                    random.uniform(-180, 180)
                ),
                capacity=capacity
            )
            warehouses.append(warehouse)
        
        return warehouses
    
    def _create_retailers(self):
        """Create retailer entities."""
        retailers = []
        
        for i in range(self.config['num_retailers']):
            retailer = Retailer(
                entity_id=f"R{i+1}",
                name=f"Retailer {i+1}",
                location=(
                    random.uniform(-90, 90),
                    random.uniform(-180, 180)
                )
            )
            retailers.append(retailer)
        
        return retailers
    
    def _setup_supply_chain_network(self):
        """Create network connections between entities."""
        self.supply_chain_network = {
            'supplier_to_manufacturer': {},
            'manufacturer_to_warehouse': {},
            'warehouse_to_retailer': {}
        }
        
        # Connect suppliers to manufacturers
        for manufacturer in self.manufacturers:
            # Each manufacturer is connected to a subset of suppliers
            connected_suppliers = random.sample(
                self.suppliers,
                k=min(len(self.suppliers), random.randint(1, 3))
            )
            self.supply_chain_network['supplier_to_manufacturer'][manufacturer.entity_id] = [
                s.entity_id for s in connected_suppliers
            ]
        
        # Connect manufacturers to warehouses
        for warehouse in self.warehouses:
            # Each warehouse is connected to a subset of manufacturers
            connected_manufacturers = random.sample(
                self.manufacturers,
                k=min(len(self.manufacturers), random.randint(1, 3))
            )
            for manufacturer in connected_manufacturers:
                if manufacturer.entity_id not in self.supply_chain_network['manufacturer_to_warehouse']:
                    self.supply_chain_network['manufacturer_to_warehouse'][manufacturer.entity_id] = []
                self.supply_chain_network['manufacturer_to_warehouse'][manufacturer.entity_id].append(
                    warehouse.entity_id
                )
        
        # Connect warehouses to retailers
        for retailer in self.retailers:
            # Each retailer is connected to a subset of warehouses
            connected_warehouses = random.sample(
                self.warehouses,
                k=min(len(self.warehouses), random.randint(1, 3))
            )
            for warehouse in connected_warehouses:
                if warehouse.entity_id not in self.supply_chain_network['warehouse_to_retailer']:
                    self.supply_chain_network['warehouse_to_retailer'][warehouse.entity_id] = []
                self.supply_chain_network['warehouse_to_retailer'][warehouse.entity_id].append(
                    retailer.entity_id
                )
    
    def _get_week_number(self):
        """Get week number of the year from current simulation date."""
        return int(self.current_date.strftime("%U"))
    
    def _log_inventory(self):
        """Record inventory levels across all entities."""
        date = self.current_date.strftime("%Y-%m-%d")
        
        # Log supplier inventory
        for supplier in self.suppliers:
            for material, capacity in supplier.capacity.items():
                log_entry = {
                    'date': date,
                    'entity_id': supplier.entity_id,
                    'entity_type': 'supplier',
                    'item_id': material,
                    'item_type': 'material',
                    'inventory_level': capacity,  # Suppliers have effectively unlimited inventory
                    'capacity': capacity
                }
                self.inventory_log.append(log_entry)
        
        # Log manufacturer inventory
        for manufacturer in self.manufacturers:
            # Raw materials
            for material, quantity in manufacturer.raw_materials.items():
                log_entry = {
                    'date': date,
                    'entity_id': manufacturer.entity_id,
                    'entity_type': 'manufacturer',
                    'item_id': material,
                    'item_type': 'material',
                    'inventory_level': quantity,
                    'capacity': None
                }
                self.inventory_log.append(log_entry)
            
            # Finished products
            for product, quantity in manufacturer.inventory.items():
                log_entry = {
                    'date': date,
                    'entity_id': manufacturer.entity_id,
                    'entity_type': 'manufacturer',
                    'item_id': product,
                    'item_type': 'product',
                    'inventory_level': quantity,
                    'capacity': manufacturer.production_capacity
                }
                self.inventory_log.append(log_entry)
        
        # Log warehouse inventory
        for warehouse in self.warehouses:
            for product, quantity in warehouse.inventory.items():
                log_entry = {
                    'date': date,
                    'entity_id': warehouse.entity_id,
                    'entity_type': 'warehouse',
                    'item_id': product,
                    'item_type': 'product',
                    'inventory_level': quantity,
                    'capacity': warehouse.total_capacity
                }
                self.inventory_log.append(log_entry)
        
        # Log retailer inventory
        for retailer in self.retailers:
            for product, quantity in retailer.inventory.items():
                log_entry = {
                    'date': date,
                    'entity_id': retailer.entity_id,
                    'entity_type': 'retailer',
                    'item_id': product,
                    'item_type': 'product',
                    'inventory_level': quantity,
                    'capacity': None
                }
                self.inventory_log.append(log_entry)
    
    def _log_transaction(self, transaction):
        """Add a transaction to the global log."""
        self.transaction_log.append({
            'date': self.current_date.strftime("%Y-%m-%d"),
            'entity_id': transaction['entity_id'],
            'transaction_type': transaction['transaction_type'],
            'product_id': transaction['product_id'],
            'quantity': transaction['quantity'],
            'partner_id': transaction['partner_id']
        })
    
    def simulate_day(self):
        """Run simulation for one day."""
        week_number = self._get_week_number()
        
        # 1. Retailers generate demand and order from warehouses
        for retailer in self.retailers:
            products_needed = {}
            
            # Check inventory and determine what products to order
            for product_id in set(p for r in self.retailers for p in r.base_demand.keys()):
                forecast = retailer.forecast_demand(product_id, week_number)
                current_stock = retailer.inventory.get(product_id, 0)
                
                # Simple reorder point calculation
                reorder_point = forecast * 2  # 2 weeks of stock
                
                if current_stock < reorder_point:
                    # Order enough to reach target stock level
                    target_stock = forecast * 4  # 4 weeks of stock
                    order_quantity = target_stock - current_stock
                    products_needed[product_id] = order_quantity
            
            # Find connected warehouses and order products
            for warehouse in self.warehouses:
                if warehouse.entity_id in self.supply_chain_network['warehouse_to_retailer'] and \
                   retailer.entity_id in self.supply_chain_network['warehouse_to_retailer'][warehouse.entity_id]:
                    
                    for product_id, quantity in products_needed.items():
                        if quantity > 0:
                            # Order from warehouse
                            shipped = warehouse.ship_to_retailer(product_id, quantity, retailer)
                            retailer.receive_shipment(product_id, shipped, warehouse)
                            products_needed[product_id] -= shipped
        
        # 2. Warehouses order from manufacturers
        for warehouse in self.warehouses:
            products_needed = {}
            
            # Check inventory and determine what to order
            for product in set(p for m in self.manufacturers for p in m.products):
                current_stock = warehouse.inventory.get(product, 0)
                # Calculate average weekly consumption
                weekly_outflow = sum(
                    t['quantity'] for t in warehouse.order_history[-30:] 
                    if t['transaction_type'] == 'ship' and t['product_id'] == product
                ) / 4 if warehouse.order_history else 20
                
                # Simple reorder point calculation
                reorder_point = weekly_outflow * 2  # 2 weeks of stock
                
                if current_stock < reorder_point:
                    # Order enough to reach target stock level
                    target_stock = weekly_outflow * 4  # 4 weeks of stock
                    order_quantity = int(target_stock - current_stock)
                    products_needed[product] = order_quantity
            
            # Find connected manufacturers and order products
            for manufacturer in self.manufacturers:
                if manufacturer.entity_id in self.supply_chain_network['manufacturer_to_warehouse'] and \
                   warehouse.entity_id in self.supply_chain_network['manufacturer_to_warehouse'][manufacturer.entity_id]:
                    
                    for product_id, quantity in products_needed.items():
                        if quantity > 0 and product_id in manufacturer.products:
                            # Order from manufacturer
                            shipped = manufacturer.ship_product(product_id, quantity, warehouse)
                            warehouse.receive_shipment(product_id, shipped, manufacturer)
                            products_needed[product_id] -= shipped
        
        # 3. Manufacturers produce products and order materials
        for manufacturer in self.manufacturers:
            # Determine production needs
            for product_id in manufacturer.products:
                # Check current inventory
                current_stock = manufacturer.inventory.get(product_id, 0)
                
                # Calculate average weekly demand
                weekly_demand = sum(
                    t['quantity'] for t in manufacturer.order_history[-30:] 
                    if t['transaction_type'] == 'ship' and t['product_id'] == product_id
                ) / 4 if manufacturer.order_history else 50
                
                # Calculate production target
                reorder_point = weekly_demand * 1.5
                if current_stock < reorder_point:
                    target_stock = weekly_demand * 3
                    production_quantity = int(target_stock - current_stock)
                    
                    # Order necessary materials first
                    if product_id in manufacturer.bom:
                        for material_id, qty_needed in manufacturer.bom[product_id].items():
                            total_needed = qty_needed * production_quantity
                            current_material = manufacturer.raw_materials.get(material_id, 0)
                            
                            if current_material < total_needed:
                                # Find supplier that can provide this material
                                for supplier in self.suppliers:
                                    if supplier.entity_id in self.supply_chain_network['supplier_to_manufacturer'].get(manufacturer.entity_id, []) and \
                                       material_id in supplier.materials:
                                        
                                        # Order with some buffer
                                        order_qty = (total_needed - current_material) * 1.2
                                        manufacturer.order_materials(supplier, material_id, int(order_qty))
                                        break
                    
                    # Attempt production
                    manufacturer.produce(product_id, production_quantity)
        
        # 4. Retailers sell to end customers
        for retailer in self.retailers:
            for product_id in set(p for r in self.retailers for p in r.base_demand.keys()):
                sale = retailer.sell_products(product_id, week_number)
                
                # Log demand
                self.demand_log.append({
                    'date': self.current_date.strftime("%Y-%m-%d"),
                    'retailer_id': retailer.entity_id,
                    'product_id': product_id,
                    'demand': sale['demand'],
                    'fulfilled': sale['sold'],
                    'stockout': sale['stockout'],
                    'stockout_rate': sale['stockout_rate']
                })
        
        # Log inventory levels
        self._log_inventory()
        
        # Move to next day
        self.current_date += self.timestep
    
    def run_simulation(self):
        """Run the full simulation from start to end date."""
        while self.current_date <= self.end_date:
            self.simulate_day()
            
            # Optional: print progress update every 7 days
            if self.current_date.day % 7 == 0:
                print(f"Simulation progressed to {self.current_date.strftime('%Y-%m-%d')}")
    
    def get_transaction_data(self):
        """Return transaction data as a DataFrame."""
        return pd.DataFrame(self.transaction_log)
    
    def get_inventory_data(self):
        """Return inventory data as a DataFrame."""
        return pd.DataFrame(self.inventory_log)
    
    def get_demand_data(self):
        """Return demand data as a DataFrame."""
        return pd.DataFrame(self.demand_log)