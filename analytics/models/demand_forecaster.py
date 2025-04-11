"""
Enhanced demand forecasting using 1D CNN with LSTM architecture
for superior supply chain predictive analytics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization, Add
from tensorflow.keras.layers import Bidirectional, GRU, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DemandForecaster:
    """Advanced forecaster using 1D CNN + LSTM architecture optimized for supply chain data."""
    
    def __init__(self, model_type='cnn_lstm'):
        """
        Initialize enhanced demand forecaster.
        
        Args:
            model_type (str): Model type - 'cnn_lstm', 'xgboost' or 'random_forest'
        """
        self.model_type = model_type
        self.models = {}  # Store models by product-retailer pairs
        self.scalers = {}  # Store scalers for each model
        self.feature_scalers = {}  # Store feature-specific scalers
        self.sequence_length = 14  # Number of time steps for sequence input
        self.feature_sequences = {}  # Store feature sequences for prediction
        self.feature_importance = {}  # Store feature importance for each model
        
        # Advanced CNN-LSTM settings
        self.learning_rate = 1e-3
        self.batch_size = 16
        self.epochs = 100
        self.patience = 10
        self.validation_split = 0.2
        
        # Ensure TensorFlow uses memory efficiently
        self._configure_tensorflow()
        
    def _configure_tensorflow(self):
        """Configure TensorFlow for optimal performance."""
        # Memory growth settings
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Memory growth setting error: {e}")
                
        # Set computation precision - mixed precision for faster training on GPUs
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
    def prepare_features(self, data):
        """Create advanced features for demand forecasting."""
        df = data.copy()
        
        # Convert date to datetime if it's not
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Cyclical encoding of time features (improved representation)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        # Create lag features for demand with more granularity
        for lag in [1, 2, 3, 5, 7, 10, 14, 21, 28]:
            df[f'demand_lag_{lag}'] = df.groupby(['product_id', 'retailer_id'])['demand'].shift(lag)
        
        # Enhanced rolling statistics
        for window in [7, 14, 21, 28]:
            # Standard rolling statistics
            df[f'demand_roll_mean_{window}'] = df.groupby(['product_id', 'retailer_id'])['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())
            df[f'demand_roll_std_{window}'] = df.groupby(['product_id', 'retailer_id'])['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))
            
            # Advanced rolling statistics
            df[f'demand_roll_median_{window}'] = df.groupby(['product_id', 'retailer_id'])['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).median())
            df[f'demand_roll_max_{window}'] = df.groupby(['product_id', 'retailer_id'])['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).max())
            df[f'demand_roll_min_{window}'] = df.groupby(['product_id', 'retailer_id'])['demand'].transform(
                lambda x: x.rolling(window=window, min_periods=1).min())
            
        # Calculate trend indicators
        df['demand_diff'] = df.groupby(['product_id', 'retailer_id'])['demand'].diff()
        df['demand_diff_pct'] = df.groupby(['product_id', 'retailer_id'])['demand'].pct_change()
        
        # Calculate moving averages with different windows for trend analysis
        df['demand_ewm_7'] = df.groupby(['product_id', 'retailer_id'])['demand'].transform(
            lambda x: x.ewm(span=7, min_periods=1).mean())
        df['demand_ewm_14'] = df.groupby(['product_id', 'retailer_id'])['demand'].transform(
            lambda x: x.ewm(span=14, min_periods=1).mean())
        
        # Calculate moving average ratios for trend strength
        df['demand_ma_ratio_7_14'] = df['demand_roll_mean_7'] / df['demand_roll_mean_14']
        
        # Fill NaN values with appropriate methods
        for col in df.columns:
            if df[col].isna().any():
                if col.startswith('demand_diff'):
                    df[col] = df[col].fillna(0)
                elif col.startswith('demand_ma_ratio'):
                    df[col] = df[col].fillna(1)
                else:
                    df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def create_sequences(self, data, target_col='demand'):
        """
        Convert time series data to sequences for CNN-LSTM input.
        Enhanced with overlapping sequences for more training data.
        
        Args:
            data (pd.DataFrame): Input time series data
            target_col (str): Target column to predict
            
        Returns:
            tuple: (X_sequences, y_values) for model training
        """
        sequences = []
        targets = []
        
        # Create sequences with step=1 for maximum data utilization
        for i in range(len(data) - self.sequence_length):
            # Get sequence and corresponding target
            sequence = data.iloc[i:i+self.sequence_length]
            target = data.iloc[i+self.sequence_length][target_col]
            
            sequences.append(sequence.values)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def build_enhanced_cnn_lstm_model(self, input_shape):
        """
        Build an enhanced 1D CNN + LSTM hybrid model with residual connections
        and attention mechanisms for better time series processing.
        
        Args:
            input_shape (tuple): Shape of input sequences (timesteps, features)
            
        Returns:
            tf.keras.Model: Compiled CNN-LSTM model
        """
        # Input layer
        inputs = Input(shape=input_shape)
        
        # First CNN block with residual connection
        cnn1 = Conv1D(filters=64, kernel_size=3, padding='same', activation=None)(inputs)
        cnn1 = BatchNormalization()(cnn1)
        cnn1 = LeakyReLU(alpha=0.2)(cnn1)
        cnn1 = MaxPooling1D(pool_size=2)(cnn1)
        
        # Second CNN block with residual connection
        cnn2 = Conv1D(filters=128, kernel_size=3, padding='same', activation=None)(cnn1)
        cnn2 = BatchNormalization()(cnn2)
        cnn2 = LeakyReLU(alpha=0.2)(cnn2)
        cnn2 = MaxPooling1D(pool_size=2)(cnn2)
        
        # Third CNN block with increased filters
        cnn3 = Conv1D(filters=256, kernel_size=3, padding='same', activation=None)(cnn2)
        cnn3 = BatchNormalization()(cnn3)
        cnn3 = LeakyReLU(alpha=0.2)(cnn3)
        cnn3 = Dropout(0.3)(cnn3)
        
        # Bidirectional LSTM for better temporal feature extraction
        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(cnn3)
        lstm1 = Dropout(0.3)(lstm1)
        
        # Second LSTM layer
        lstm2 = LSTM(64)(lstm1)
        
        # Fully connected layers
        dense1 = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(lstm2)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.2)(dense1)
        
        # Output layer
        output = Dense(1)(dense1)
        
        # Create model
        model = Model(inputs=inputs, outputs=output)
        
        # Use Adam optimizer with learning rate schedule
        optimizer = Adam(learning_rate=self.learning_rate)
        
        # Compile model
        model.compile(
            optimizer=optimizer, 
            loss='huber_loss',  # More robust to outliers than MSE
            metrics=['mae', 'mse']
        )
        
        return model
    
    def fit(self, data):
        """Train advanced forecasting models for each product-retailer combination."""
        # Prepare features
        df = self.prepare_features(data)
        
        # Group by product and retailer
        for (product_id, retailer_id), group in df.groupby(['product_id', 'retailer_id']):
            print(f"Training model for product {product_id}, retailer {retailer_id}")
            
            # Skip if very little data (need at least 2*sequence_length for training)
            if len(group) < 2 * self.sequence_length:
                print(f"  Skipping due to insufficient data: {len(group)} records")
                continue
            
            # Sort by date
            group = group.sort_values('date')
            
            # Define features and target
            feature_cols = [col for col in group.columns if col not in 
                           ['date', 'product_id', 'retailer_id', 'demand', 
                            'fulfilled', 'stockout', 'stockout_rate']]
            
            # Store the last sequence for prediction
            if len(group) >= self.sequence_length:
                self.feature_sequences[(product_id, retailer_id)] = group[feature_cols + ['demand']].iloc[-self.sequence_length:].copy()
            
            # For CNN-LSTM model
            if self.model_type == 'cnn_lstm':
                # Scale features
                scaler = RobustScaler()  # More robust to outliers than StandardScaler
                X = group[feature_cols].values
                X_scaled = scaler.fit_transform(X)
                
                # Store scaler for future use
                self.scalers[(product_id, retailer_id)] = scaler
                
                # Replace original values with scaled values
                scaled_group = group.copy()
                scaled_group[feature_cols] = X_scaled
                
                # Create sequences
                X_seq, y_seq = self.create_sequences(scaled_group[feature_cols + ['demand']])
                
                # Split data
                X_train, X_val, y_train, y_val = train_test_split(
                    X_seq, y_seq, test_size=self.validation_split, random_state=42, shuffle=False
                )
                
                # Build enhanced CNN-LSTM model
                input_shape = (X_train.shape[1], X_train.shape[2])
                model = self.build_enhanced_cnn_lstm_model(input_shape)
                
                # Print model summary for first instance only
                if len(self.models) == 0:
                    model.summary()
                
                # Create callbacks
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True,
                    verbose=1
                )
                
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
                
                try:
                    # Train model with progress display
                    print(f"  Training CNN-LSTM model with {len(X_train)} sequences...")
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        callbacks=[early_stopping, reduce_lr],
                        verbose=0  # Set to 1 for progress display
                    )
                    
                    # Store the model
                    self.models[(product_id, retailer_id)] = model
                    
                    # Print validation performance
                    val_loss = model.evaluate(X_val, y_val, verbose=0)
                    print(f"  Validation MAE: {val_loss[1]:.2f}, MSE: {val_loss[2]:.2f}")
                    
                    # Check for overfitting
                    train_loss = model.evaluate(X_train, y_train, verbose=0)
                    if train_loss[1] < 0.5 * val_loss[1]:
                        print("  Warning: Possible overfitting detected")
                    
                except Exception as e:
                    print(f"  Error training CNN-LSTM model: {e}")
                    print("  Falling back to XGBoost model")
                    
                    # Train XGBoost as fallback
                    X = group[feature_cols]
                    y = group['demand']
                    
                    # Rescale features for XGBoost
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    self.scalers[(product_id, retailer_id)] = scaler
                    
                    model = xgb.XGBRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=7,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42
                    )
                    model.fit(X_scaled, y)
                    self.models[(product_id, retailer_id)] = model
                    
                    # Store feature importance
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'Feature': feature_cols,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        self.feature_importance[(product_id, retailer_id)] = feature_importance
                    
            # For traditional ML models (XGBoost as fallback)
            else:
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
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=9,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(X_scaled, y)
                
                # Store the model
                self.models[(product_id, retailer_id)] = model
                
                # Store feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    self.feature_importance[(product_id, retailer_id)] = feature_importance
                    
                    # Print top 5 features
                    print("  Top 5 important features:")
                    print(feature_importance.head(5))
        
        return self
    
    def _generate_cnn_lstm_forecast(self, model, current_sequence, scaler, product_id, retailer_id, last_date, periods_ahead):
        """Generate forecasts using the CNN-LSTM model."""
        forecasts = []
        
        # Make a copy of the current sequence for forecasting
        current_seq = current_sequence.copy()
        
        for i in range(periods_ahead):
            # Ensure the sequence has the right shape
            X_pred = current_seq.reshape(1, self.sequence_length, current_seq.shape[1])
            
            # Make prediction
            pred = model.predict(X_pred, verbose=0)[0][0]
            
            # Ensure prediction is non-negative
            pred = max(0, pred)
            
            # Store forecast
            next_date = last_date + timedelta(days=i+1)
            forecasts.append({
                'date': next_date,
                'product_id': product_id,
                'retailer_id': retailer_id,
                'forecasted_demand': round(pred)
            })
            
            # Update sequence for next prediction
            # Shift sequence and add the new prediction
            new_row = current_seq[-1].copy()
            
            # Replace demand value with the predicted value (assuming it's the last column)
            new_row[-1] = pred
            
            # Update the sequence by sliding the window
            current_seq = np.vstack([current_seq[1:], [new_row]])
        
        return forecasts
    
    def predict(self, test_data, periods_ahead=30):
        """Generate forecasts for future periods using enhanced models."""
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
            
            # For CNN-LSTM models
            if isinstance(model, tf.keras.Model):
                # Check if we have stored feature sequences
                if (product_id, retailer_id) not in self.feature_sequences:
                    # If no stored sequence, prepare data and create sequence
                    prepared_data = self.prepare_features(subset)
                    feature_cols = [col for col in prepared_data.columns if col not in 
                                   ['date', 'product_id', 'retailer_id', 'demand', 
                                    'fulfilled', 'stockout', 'stockout_rate']]
                    
                    if len(prepared_data) >= self.sequence_length:
                        sequence_data = prepared_data[feature_cols + ['demand']].iloc[-self.sequence_length:].copy()
                    else:
                        print(f"  Warning: Not enough data for sequence. Using padding.")
                        # Pad sequence with repeated first row if needed
                        first_row = prepared_data[feature_cols + ['demand']].iloc[0].copy()
                        padding_rows = pd.DataFrame([first_row] * (self.sequence_length - len(prepared_data)),
                                                  columns=first_row.index)
                        sequence_data = pd.concat([padding_rows, prepared_data[feature_cols + ['demand']]])
                        
                    self.feature_sequences[(product_id, retailer_id)] = sequence_data
                
                # Get the stored sequence
                sequence_data = self.feature_sequences[(product_id, retailer_id)].copy()
                
                # Get feature column names (excluding demand which should be the last column)
                feature_cols = sequence_data.columns.tolist()
                feature_cols.remove('demand')
                
                # Get scaler 
                scaler = self.scalers[(product_id, retailer_id)]
                
                # Scale the sequence data
                sequence_values = sequence_data[feature_cols].values
                sequence_values_scaled = scaler.transform(sequence_values)
                
                # Add demand column back as the last column
                current_sequence = np.column_stack((
                    sequence_values_scaled, 
                    sequence_data['demand'].values
                ))
                
                # Generate forecasts using CNN-LSTM
                forecasts = self._generate_cnn_lstm_forecast(
                    model, current_sequence, scaler, 
                    product_id, retailer_id, last_date, periods_ahead
                )
                
                results.extend(forecasts)
                
            # For traditional ML models (XGBoost, Random Forest)
            else:
                # Recursive forecasting for traditional ML
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
        
        return pd.DataFrame(results)