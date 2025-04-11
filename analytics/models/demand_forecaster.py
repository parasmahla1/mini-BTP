"""
Demand forecasting models for supply chain predictive analytics.
Using 1D CNN with LSTM architecture for improved performance.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization

class DemandForecaster:
    """Forecasts future demand based on historical patterns using 1D CNN + LSTM."""
    
    def __init__(self, model_type='cnn_lstm'):
        """
        Initialize demand forecaster.
        
        Args:
            model_type (str): Model type - 'cnn_lstm', 'xgboost' or 'random_forest'
        """
        self.model_type = model_type
        self.models = {}  # Store models by product-retailer pairs
        self.scalers = {}  # Store scalers for each model
        self.sequence_length = 14  # Number of time steps to use for sequence input to CNN-LSTM
        self.feature_sequences = {}  # Store feature sequences for prediction
        
    def prepare_features(self, data):
        """Create features for demand forecasting."""
        df = data.copy()
        
        # Convert date to datetime if it's not
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Cyclical encoding of time features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Create lag features for demand
        for lag in [1, 2, 3, 7, 14]:
            df[f'demand_lag_{lag}'] = df.groupby(['product_id', 'retailer_id'])['demand'].shift(lag)
        
        # Create rolling statistics
        for window in [7, 14, 28]:
            df[f'demand_rolling_mean_{window}'] = df.groupby(['product_id', 'retailer_id'])['demand'].transform(
                lambda x: x.rolling(window=window).mean().shift(1))
            df[f'demand_rolling_std_{window}'] = df.groupby(['product_id', 'retailer_id'])['demand'].transform(
                lambda x: x.rolling(window=window).std().shift(1))
        
        # Drop NaN values from lag features
        df = df.dropna()
        
        return df
    
    def create_sequences(self, data, target_col='demand'):
        """
        Convert time series data to sequences for CNN-LSTM input.
        
        Args:
            data (pd.DataFrame): Input time series data
            target_col (str): Target column to predict
            
        Returns:
            tuple: (X_sequences, y_values) for model training
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            sequence = data.iloc[i:i+self.sequence_length]
            target = data.iloc[i+self.sequence_length][target_col]
            
            sequences.append(sequence.values)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def build_cnn_lstm_model(self, input_shape):
        """
        Build a 1D CNN + LSTM hybrid model.
        
        Args:
            input_shape (tuple): Shape of input sequences (timesteps, features)
            
        Returns:
            tf.keras.Model: Compiled CNN-LSTM model
        """
        model = Sequential()
        
        # 1D CNN layers for feature extraction
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # LSTM layers for temporal dependencies
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(50))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def fit(self, data):
        """Train forecasting models for each product-retailer combination."""
        # Prepare features
        df = self.prepare_features(data)
        
        # Group by product and retailer
        for (product_id, retailer_id), group in df.groupby(['product_id', 'retailer_id']):
            print(f"Training model for product {product_id}, retailer {retailer_id}")
            
            # Sort by date
            group = group.sort_values('date')
            
            # Define features and target
            feature_cols = [col for col in group.columns if col not in 
                           ['date', 'product_id', 'retailer_id', 'demand', 
                            'fulfilled', 'stockout', 'stockout_rate']]
            
            # Store the last sequence for prediction
            if len(group) >= self.sequence_length:
                self.feature_sequences[(product_id, retailer_id)] = group[feature_cols].iloc[-self.sequence_length:].copy()
            
            # For traditional ML models (XGBoost, Random Forest)
            if self.model_type in ['xgboost', 'random_forest']:
                X = group[feature_cols]
                y = group['demand']
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Save scaler
                self.scalers[(product_id, retailer_id)] = scaler
                
                # Train model
                if self.model_type == 'xgboost':
                    model = xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=7,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42
                    )
                    model.fit(X_scaled, y)
                else:  # random_forest
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=12,
                        min_samples_split=5,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(X_scaled, y)
                
                # Store the model
                self.models[(product_id, retailer_id)] = model
                
            # For CNN-LSTM model
            elif self.model_type == 'cnn_lstm':
                # Need sufficient data for sequence creation
                if len(group) <= self.sequence_length + 10:
                    print(f"  Warning: Not enough data for CNN-LSTM model. Defaulting to XGBoost.")
                    # Fall back to XGBoost
                    X = group[feature_cols]
                    y = group['demand']
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Save scaler
                    self.scalers[(product_id, retailer_id)] = scaler
                    
                    model = xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=7,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42
                    )
                    model.fit(X_scaled, y)
                    self.models[(product_id, retailer_id)] = model
                else:
                    # Scale all features for CNN-LSTM
                    X = group[feature_cols].values
                    scaler = MinMaxScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Save scaler
                    self.scalers[(product_id, retailer_id)] = scaler
                    
                    # Replace original values with scaled values
                    scaled_group = group.copy()
                    scaled_group[feature_cols] = X_scaled
                    
                    # Create sequences
                    X_seq, y_seq = self.create_sequences(scaled_group[feature_cols + ['demand']])
                    
                    # Split data
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
                    )
                    
                    # Build and train CNN-LSTM model
                    input_shape = (X_train.shape[1], X_train.shape[2])
                    model = self.build_cnn_lstm_model(input_shape)
                    
                    # Early stopping to prevent overfitting
                    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                    
                    try:
                        model.fit(
                            X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=50,
                            batch_size=32,
                            callbacks=[early_stopping],
                            verbose=0
                        )
                        
                        # Store the model
                        self.models[(product_id, retailer_id)] = model
                        
                        # Print validation performance
                        val_loss = model.evaluate(X_val, y_val, verbose=0)
                        print(f"  Validation MAE: {val_loss[1]:.2f}")
                        
                    except Exception as e:
                        print(f"  Error training CNN-LSTM model: {e}")
                        print("  Falling back to XGBoost model")
                        
                        # Fall back to XGBoost
                        X = group[feature_cols]
                        y = group['demand']
                        
                        # Rescale features for XGBoost
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        self.scalers[(product_id, retailer_id)] = scaler
                        
                        model = xgb.XGBRegressor(
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=7,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42
                        )
                        model.fit(X_scaled, y)
                        self.models[(product_id, retailer_id)] = model
        
        return self
    
    def predict(self, test_data, periods_ahead=30):
        """Generate forecasts for future periods."""
        results = []
        
        # Process each product-retailer combination
        for (product_id, retailer_id), model in self.models.items():
            # Filter data for this product-retailer
            subset = test_data[(test_data['product_id'] == product_id) & 
                              (test_data['retailer_id'] == retailer_id)]
            
            if len(subset) == 0:
                continue
                
            # Sort by date
            subset = subset.sort_values('date')
            
            # Get the last date
            last_date = subset['date'].max()
            
            # For traditional ML models (XGBoost, Random Forest)
            if not isinstance(model, tf.keras.Model):
                # Recursive forecasting
                forecasts = []
                current_data = subset.copy()
                
                for i in range(periods_ahead):
                    # Create next day's row
                    next_date = last_date + timedelta(days=i+1)
                    next_row = pd.DataFrame({'date': [next_date], 
                                          'product_id': [product_id],
                                          'retailer_id': [retailer_id]})
                    
                    # Add to current data
                    temp_data = pd.concat([current_data, next_row])
                    
                    # Prepare features
                    temp_data = self.prepare_features(temp_data)
                    
                    # Get features for the new row
                    new_features = temp_data.iloc[-1:]
                    feature_cols = [col for col in new_features.columns if col not in 
                                  ['date', 'product_id', 'retailer_id', 'demand', 
                                   'fulfilled', 'stockout', 'stockout_rate']]
                    X_new = new_features[feature_cols]
                    
                    # Scale features
                    scaler = self.scalers[(product_id, retailer_id)]
                    X_new_scaled = scaler.transform(X_new)
                    
                    # Make prediction
                    pred = model.predict(X_new_scaled)[0]
                    
                    # Store forecast
                    forecasts.append({
                        'date': next_date,
                        'product_id': product_id,
                        'retailer_id': retailer_id,
                        'forecasted_demand': max(0, round(pred))
                    })
                    
                    # Update for next iteration: add predicted demand
                    next_row['demand'] = max(0, round(pred))
                    current_data = pd.concat([current_data, next_row])
                
                results.extend(forecasts)
                
            # For CNN-LSTM model
            else:
                # Check if we have stored feature sequences
                if (product_id, retailer_id) not in self.feature_sequences:
                    # Prepare data
                    prepared_data = self.prepare_features(subset)
                    feature_cols = [col for col in prepared_data.columns if col not in 
                                   ['date', 'product_id', 'retailer_id', 'demand', 
                                    'fulfilled', 'stockout', 'stockout_rate']]
                    
                    # Store the last sequence
                    self.feature_sequences[(product_id, retailer_id)] = prepared_data[feature_cols].iloc[-self.sequence_length:].copy()
                
                # Get the stored sequence
                last_sequence = self.feature_sequences[(product_id, retailer_id)].copy()
                
                # Get scaler 
                scaler = self.scalers[(product_id, retailer_id)]
                
                # Scale the sequence
                last_sequence_scaled = scaler.transform(last_sequence)
                
                # Prepare forecasts
                forecasts = []
                current_sequence = last_sequence_scaled.copy()
                
                for i in range(periods_ahead):
                    # Reshape sequence for model input
                    X_pred = current_sequence.reshape(1, self.sequence_length, current_sequence.shape[1])
                    
                    # Make prediction
                    pred = model.predict(X_pred, verbose=0)[0][0]
                    
                    # Store forecast
                    next_date = last_date + timedelta(days=i+1)
                    forecasts.append({
                        'date': next_date,
                        'product_id': product_id,
                        'retailer_id': retailer_id,
                        'forecasted_demand': max(0, round(pred))
                    })
                    
                    # Generate features for the new prediction
                    # This is simplified - ideally we would properly generate all features
                    # For now, we'll shift the sequence and add the new prediction
                    new_row = current_sequence[-1].copy()
                    # Assuming the target is the last feature in our sequence
                    new_row[-1] = pred
                    
                    # Update sequence for next prediction (shift and add new prediction)
                    current_sequence = np.vstack([current_sequence[1:], new_row])
                
                results.extend(forecasts)
        
        return pd.DataFrame(results)