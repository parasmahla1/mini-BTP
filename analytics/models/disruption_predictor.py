"""
Disruption prediction models for supply chain risk management.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class DisruptionPredictor:
    """Predicts supply chain disruptions based on historical patterns."""
    
    def __init__(self):
        """Initialize the disruption predictor."""
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def prepare_features(self, data):
        """Prepare features for disruption prediction."""
        df = data.copy()
        
        # Check if any essential columns are missing
        required_columns = ['entity_id', 'item_id', 'inventory_level']
        if not all(col in df.columns for col in required_columns):
            print(f"Warning: Missing required columns for disruption prediction. Available columns: {df.columns}")
            missing_cols = [col for col in required_columns if col not in df.columns]
            print(f"Missing: {missing_cols}")
            return pd.DataFrame()  # Return empty DataFrame if missing essential columns
        
        # Define what constitutes a disruption
        # Example: stockout rate > 20% or delivery delay > 3 days
        if 'stockout_rate' in df.columns:
            df['disruption'] = (df['stockout_rate'] > 0.2).astype(int)
        elif 'lead_time' in df.columns and 'expected_lead_time' in df.columns:
            df['disruption'] = ((df['lead_time'] - df['expected_lead_time']) > 3).astype(int)
        else:
            # Create synthetic disruption indicator for simulation
            if 'inventory_level' in df.columns and 'item_type' in df.columns:
                df['disruption'] = ((df['inventory_level'] == 0) & 
                                (df['item_type'] == 'product')).astype(int)
            else:
                df['disruption'] = (df['inventory_level'] == 0).astype(int)
        
        # Create features
        # 1. Inventory levels and trends
        if 'inventory_level' in df.columns:
            if 'capacity' in df.columns:
                df['inventory_ratio'] = df['inventory_level'] / df['capacity'].fillna(1)
            else:
                df['inventory_ratio'] = df['inventory_level'] / df['inventory_level'].max()
                
            # Calculate inventory trends (requires time series data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                # 7-day rolling average
                df['inventory_7d_avg'] = df.groupby(['entity_id', 'item_id'])['inventory_level'].transform(
                    lambda x: x.rolling(7, min_periods=1).mean())
                
                # Inventory trend (increasing/decreasing)
                df['inventory_trend'] = df.groupby(['entity_id', 'item_id'])['inventory_level'].diff()
        
        # 2. Transaction patterns - skip if transaction_type column doesn't exist
        
        # 3. Time-based features
        if 'date' in df.columns:
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
        
        # Drop rows with missing values
        df = df.dropna()
        
        return df
    
    def fit(self, data):
        """Train the disruption prediction model."""
        # Prepare features
        df = self.prepare_features(data)
        
        # Define features and target
        feature_cols = [col for col in df.columns if col not in 
                       ['date', 'entity_id', 'entity_type', 'item_id', 'item_type', 
                        'disruption', 'stockout_rate', 'lead_time']]
        
        X = df[feature_cols]
        y = df['disruption']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        print("Disruption Prediction Model Performance:")
        print(f"Accuracy: {report['accuracy']:.2f}")
        print(f"Precision: {report['1']['precision']:.2f}")
        print(f"Recall: {report['1']['recall']:.2f}")
        print(f"F1-Score: {report['1']['f1-score']:.2f}")
        
        return self
    
    def predict_disruptions(self, data, threshold=0.5):
        """Predict probability of disruptions."""
        # Prepare features
        df = self.prepare_features(data)
        
        # Define features
        feature_cols = [col for col in df.columns if col not in 
                       ['date', 'entity_id', 'entity_type', 'item_id', 'item_type', 
                        'disruption', 'stockout_rate', 'lead_time']]
        
        # Check if we have the required features
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        X = df[feature_cols]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict disruption probabilities
        disruption_probs = self.model.predict_proba(X_scaled)[:, 1]
        
        # Add predictions to dataframe
        results = df[['date', 'entity_id', 'item_id']].copy()
        results['disruption_probability'] = disruption_probs
        results['predicted_disruption'] = (disruption_probs >= threshold).astype(int)
        
        return results