"""
Lead time prediction models for supply chain analytics.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

class LeadTimeEstimator:
    """Predicts lead times for supply chain transactions."""
    
    def __init__(self):
        """Initialize the lead time estimator."""
        self.model = None
        self.preprocessor = None
        
    def prepare_features(self, data):
        """Prepare features for lead time prediction."""
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        if 'lead_time' not in df.columns:
            print("Warning: No lead_time column found in the data.")
            return df
        
        # Extract useful features
        features = ['entity_id', 'partner_id', 'product_id']
        
        # Add transaction type if available
        if 'transaction_type' in df.columns:
            features.append('transaction_type')
        
        # Add quantity if available
        if 'quantity' in df.columns:
            features.append('quantity')
        
        # Add time-based features if date is available
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            features.extend(['day_of_week', 'month'])
        
        # Return features and target
        return df[features + ['lead_time']]
    
    def fit(self, data):
        """Train the lead time prediction model."""
        # Prepare the dataset
        df = self.prepare_features(data)
        
        if len(df) == 0 or 'lead_time' not in df.columns:
            print("Error: No valid data for lead time estimation.")
            return self
        
        # Define features and target
        X = df.drop('lead_time', axis=1)
        y = df['lead_time']
        
        # Identify categorical and numerical features
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessor
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create and train the model
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ])
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print("Lead Time Estimator Performance:")
        print(f"MAE: {mae:.2f} days")
        print(f"RMSE: {rmse:.2f} days")
        
        return self
    
    def predict(self, data):
        """Predict lead times for new transactions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Prepare features
        df = self.prepare_features(data)
        
        if 'lead_time' in df.columns:
            df = df.drop('lead_time', axis=1)
        
        # Make predictions
        predictions = self.model.predict(df)
        
        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 1)
        
        return predictions