"""
DemandForecaster class for predicting demand in the supply chain.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv1D, LSTM, Dense, Dropout, BatchNormalization,
                                   Input, Concatenate, Bidirectional, Flatten)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class DemandForecaster:
    """
    DemandForecaster using deep learning models for time series forecasting.
    """
    
    def __init__(self, model_type='cnn_lstm', sequence_length=14, batch_size=32, epochs=100, use_ensemble=False):
        """
        Initialize the demand forecaster with configurable parameters.
        
        Args:
            model_type: Type of model architecture ('cnn_lstm', 'lstm', 'xgboost', 'random_forest')
            sequence_length: Number of time steps to use as input
            batch_size: Training batch size
            epochs: Maximum training epochs
            use_ensemble: Whether to use ensemble of models
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_ensemble = use_ensemble
        self.forecast_horizon = 7  # Default forecast horizon (1 week)
        
        # Initialize scalers
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Will be initialized during training
        self.model = None
        self.history = None
        self.feature_cols = []
        self.target_col = 'demand'  # Default target column name
        
        # For ensemble models
        self.models = []
        self.weights = []
    
    def _preprocess_data(self, data):
        """
        Preprocess data for model training.
        
        Args:
            data: Pandas DataFrame with time series data
            
        Returns:
            Processed data with additional features
        """
        # Copy to avoid modifying original
        df = data.copy()
        
        # Ensure data is sorted by date
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        # Check if the target column exists, if not try to use others
        if 'demand' not in df.columns:
            if 'actual_demand' in df.columns:
                df['demand'] = df['actual_demand']
            elif 'fulfilled' in df.columns:
                df['demand'] = df['fulfilled']
                
        self.target_col = 'demand'  # Ensure target column is set
        
        # Extract temporal features
        if 'date' in df.columns:
            df['dayofweek'] = pd.to_datetime(df['date']).dt.dayofweek
            df['month'] = pd.to_datetime(df['date']).dt.month
            
            # One-hot encode categorical time features
            df = pd.get_dummies(df, columns=['dayofweek', 'month'])
        
        # Fill missing values
        df = df.fillna(method='ffill').fillna(0)
        
        # Add lagged features (only for deep learning models)
        if self.model_type in ['cnn_lstm', 'lstm']:
            # Add lags if we have enough data
            min_periods = min(5, max(1, len(df) // 10))
            for lag in range(1, min_periods + 1):
                df[f'demand_lag_{lag}'] = df[self.target_col].shift(lag)
            
            # Add rolling statistics if we have enough data
            if len(df) >= 7:
                df['demand_rolling_mean_7d'] = df[self.target_col].rolling(window=7, min_periods=1).mean()
                
            # Fill any remaining NAs from lag creation
            df = df.fillna(0)
        
        return df
    
    def _create_sequences(self, data, is_training=True):
        """
        Create sequences for time series forecasting.
        
        Args:
            data: Preprocessed DataFrame
            is_training: Whether this is for training (includes target creation)
            
        Returns:
            X, y arrays for model training or prediction
        """
        # Identify feature columns (exclude date and target)
        if not self.feature_cols:
            self.feature_cols = [c for c in data.columns 
                               if c != 'date' and c != self.target_col and c != 'product_id' 
                               and c != 'retailer_id' and c != 'entity_id']
        
        # Extract features and target
        features = data[self.feature_cols].values
        targets = data[self.target_col].values if self.target_col in data.columns else None
        
        # Scale features
        if is_training:
            features = self.scaler_x.fit_transform(features)
            if targets is not None:
                targets = self.scaler_y.fit_transform(targets.reshape(-1, 1)).flatten()
        else:
            features = self.scaler_x.transform(features)
            if targets is not None:
                targets = self.scaler_y.transform(targets.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = [], []
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            X.append(features[i:i+self.sequence_length])
            if is_training and targets is not None:
                y.append(targets[i+self.sequence_length:i+self.sequence_length+self.forecast_horizon])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y) if y else None
        
        return X, y
    
    def _build_cnn_lstm_model(self, input_shape):
        """
        Build the CNN-LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled keras model
        """
        # Input layer
        inputs = Input(shape=input_shape)
        
        # CNN layers for feature extraction
        x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        # LSTM layer for temporal dependencies
        x = LSTM(128, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        
        # Dense layers for output
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.forecast_horizon)(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def _build_lstm_model(self, input_shape):
        """
        Build a simple LSTM model.
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Compiled keras model
        """
        model = Sequential()
        model.add(LSTM(100, activation='relu', input_shape=input_shape))
        model.add(Dense(self.forecast_horizon))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def fit(self, data):
        """
        Train the model on the provided data.
        
        Args:
            data: DataFrame with historical demand data
            
        Returns:
            Self for method chaining
        """
        # Handle empty dataframe
        if data.empty:
            print("Empty dataframe provided. Cannot train model.")
            return self
        
        # Preprocess data
        processed_data = self._preprocess_data(data)
        
        # For non-deep learning models, handle differently
        if self.model_type in ['xgboost', 'random_forest']:
            print(f"{self.model_type} not implemented yet. Using CNN-LSTM instead.")
            self.model_type = 'cnn_lstm'
        
        # Create sequences for deep learning models
        X, y = self._create_sequences(processed_data)
        
        # Skip training if we don't have enough data
        if X.shape[0] == 0 or y.shape[0] == 0:
            print(f"Not enough data to train. X shape: {X.shape}, y shape: {y.shape}")
            return self
        
        # Split data for validation
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Build and train model based on type
        if self.model_type == 'cnn_lstm':
            self.model = self._build_cnn_lstm_model((X.shape[1], X.shape[2]))
        elif self.model_type == 'lstm':
            self.model = self._build_lstm_model((X.shape[1], X.shape[2]))
        
        # Print model summary
        self.model.summary()
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self
    
    def predict(self, data, periods_ahead=None, return_conf_int=False):
        """
        Generate forecasts for future periods.
        
        Args:
            data: DataFrame with recent demand data
            periods_ahead: Number of periods to forecast (default uses model's horizon)
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            DataFrame with forecasts
        """
        if self.model is None:
            print("Model not trained. Please call fit() before predict().")
            # Return empty dataframe with expected columns
            return pd.DataFrame(columns=['date', 'forecast'])
            
        if periods_ahead is None:
            periods_ahead = self.forecast_horizon
        
        # Preprocess data
        processed_data = self._preprocess_data(data)
        
        # Create sequences for prediction
        X, _ = self._create_sequences(processed_data, is_training=False)
        
        if len(X) == 0:
            print("Not enough data to create prediction sequences")
            # Return empty dataframe with expected columns
            return pd.DataFrame(columns=['date', 'forecast'])
        
        # Generate predictions
        predictions = self.model.predict(X)
        
        # Inverse transform predictions to original scale
        predictions = self.scaler_y.inverse_transform(predictions)
        
        # Create forecast DataFrame
        last_date = pd.to_datetime(data['date'].iloc[-1])
        dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods_ahead,
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'date': dates,
            'forecast': predictions[-1]  # Use the most recent prediction
        })
        
        # Add product and retailer IDs if they exist in original data
        for col in ['product_id', 'retailer_id', 'entity_id']:
            if col in data.columns:
                forecast_df[col] = data[col].iloc[-1]
        
        # Add confidence intervals if requested (simple method)
        if return_conf_int:
            std_dev = 0.1 * predictions[-1]  # Simple approximation
            forecast_df['lower_bound'] = predictions[-1] - 2 * std_dev
            forecast_df['upper_bound'] = predictions[-1] + 2 * std_dev
            forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(0)  # No negative demand
        
        return forecast_df
    
    def evaluate(self, test_data):
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: DataFrame with test data
            
        Returns:
            Dictionary of performance metrics
        """
        if self.model is None:
            print("Model not trained. Please call fit() before evaluate().")
            return {'error': 'Model not trained'}
            
        # Preprocess data
        processed_data = self._preprocess_data(test_data)
        
        # Create sequences
        X_test, y_test = self._create_sequences(processed_data)
        
        if len(X_test) == 0 or len(y_test) == 0:
            print("Not enough test data to evaluate")
            return {'error': 'Not enough test data'}
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        
        # Inverse transform predictions and actual values
        y_pred = self.scaler_y.inverse_transform(y_pred)
        y_true = self.scaler_y.inverse_transform(y_test)
        
        # Compute metrics
        metrics = {}
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        
        # Calculate MAPE safely
        mask = y_true > 0  # Avoid division by zero
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics['mape'] = float(mape)
        else:
            metrics['mape'] = float('nan')
        
        return metrics
    
    def save_model(self, filepath):
        """Save the model to file."""
        if self.model is None:
            print("Model not trained. Please call fit() before save_model().")
            return
            
        # Save model
        self.model.save(filepath)
        
        # Save scalers and parameters
        import joblib
        import os
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save parameters
        params = {
            'model_type': self.model_type,
            'sequence_length': self.sequence_length,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'forecast_horizon': self.forecast_horizon,
            'feature_cols': self.feature_cols,
            'target_col': self.target_col
        }
        
        joblib.dump(params, filepath + '_params.pkl')
        joblib.dump(self.scaler_x, filepath + '_scaler_x.pkl')
        joblib.dump(self.scaler_y, filepath + '_scaler_y.pkl')