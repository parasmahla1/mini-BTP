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
import psutil
import time
from datetime import datetime
import os

class DemandForecaster:
    """
    DemandForecaster using deep learning models for time series forecasting.
    """
    def _setup_gpu(self):
        """Configure TensorFlow to use GPU if available."""

    
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Configure TensorFlow to use the first GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"GPU setup complete: Found {len(gpus)} GPU(s)")
            
            # Use mixed precision for faster training
            if hasattr(tf.keras.mixed_precision, 'set_global_policy'):
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("Using mixed precision training (float16)")
            
            
            
        except Exception as e:
            print(f"Error configuring GPU: {e}")
    
    print("No GPU detected. Using CPU instead.")
        
    def __init__(self, model_type='cnn_lstm', sequence_length=14, batch_size=32, 
                 epochs=100, use_ensemble=False, model_size='large'):
        """
        Initialize the demand forecaster with configurable parameters.
        
        Args:
            model_type: Type of model architecture ('cnn_lstm', 'lstm', 'xgboost', 'random_forest')
            sequence_length: Number of time steps to use as input
            batch_size: Training batch size
            epochs: Maximum training epochs
            use_ensemble: Whether to use ensemble of models
            model_size: Model capacity ('small', 'medium', 'large', 'xlarge')
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_ensemble = use_ensemble
        self.model_size = model_size
        self.forecast_horizon = 7  # Default forecast horizon (1 week)
        
        # Setup GPU if available
        self._setup_gpu()
        
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
            df['quarter'] = pd.to_datetime(df['date']).dt.quarter
            df['year'] = pd.to_datetime(df['date']).dt.year
            
            # One-hot encode categorical time features
            df = pd.get_dummies(df, columns=['dayofweek', 'month', 'quarter'])
        
        # Fill missing values - use ffill and bfill instead of method='ffill'
        df = df.ffill().fillna(0)
        
        # Add lagged features (only for deep learning models)
        if self.model_type in ['cnn_lstm', 'lstm']:
            # Add lags if we have enough data
            min_periods = min(7, max(1, len(df) // 10))
            for lag in range(1, min_periods + 1):
                df[f'demand_lag_{lag}'] = df[self.target_col].shift(lag)
            
            # Add rolling statistics if we have enough data
            if len(df) >= 7:
                df['demand_rolling_mean_7d'] = df[self.target_col].rolling(window=7, min_periods=1).mean()
                df['demand_rolling_std_7d'] = df[self.target_col].rolling(window=7, min_periods=1).std().fillna(0)
                
            if len(df) >= 14:
                df['demand_rolling_mean_14d'] = df[self.target_col].rolling(window=14, min_periods=1).mean()
                
            # Add trend features
            df['demand_diff_1d'] = df[self.target_col].diff().fillna(0)
            df['demand_diff_7d'] = df[self.target_col].diff(7).fillna(0)
                
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
        Build an enhanced CNN-LSTM model architecture with ~1M parameters.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled keras model
        """
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Adjust architecture based on model size
        if self.model_size == 'small':
            # Small model (~100K parameters)
            conv1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
            conv1 = BatchNormalization()(conv1)
            
            lstm1 = LSTM(64)(conv1)
            dense1 = Dense(32, activation='relu')(lstm1)
            outputs = Dense(self.forecast_horizon)(dense1)
            
        elif self.model_size == 'medium':
            # Medium model (~250K parameters)
            conv1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
            conv1 = BatchNormalization()(conv1)
            
            conv2 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv1)
            conv2 = BatchNormalization()(conv2)
            
            lstm1 = LSTM(128)(conv2)
            dense1 = Dense(64, activation='relu')(lstm1)
            outputs = Dense(self.forecast_horizon)(dense1)
            
        elif self.model_size == 'large':
            # Large model (~500K-1M parameters)
            # Multi-scale CNN feature extraction with multiple filter sizes
            conv1_1 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')(inputs)
            conv1_2 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(inputs)
            conv1 = Concatenate()([conv1_1, conv1_2])
            conv1 = BatchNormalization()(conv1)
            
            conv2 = Conv1D(filters=192, kernel_size=3, padding='same', activation='relu')(conv1)
            conv2 = BatchNormalization()(conv2)
            
            lstm1 = Bidirectional(LSTM(192))(conv2)
            lstm1 = Dropout(0.3)(lstm1)
            
            dense1 = Dense(256, activation='relu')(lstm1)
            dense1 = BatchNormalization()(dense1)
            dense1 = Dropout(0.3)(dense1)
            
            dense2 = Dense(128, activation='relu')(dense1)
            outputs = Dense(self.forecast_horizon)(dense2)
            
        else:  # xlarge
            # XLarge model (>1M parameters)
            # Multi-scale CNN feature extraction with multiple filter sizes
            conv1_1 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')(inputs)
            conv1_2 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(inputs)
            conv1_3 = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(inputs)
            conv1 = Concatenate()([conv1_1, conv1_2, conv1_3])  # Combine different receptive fields
            conv1 = BatchNormalization()(conv1)
            
            # Second convolutional block
            conv2_1 = Conv1D(filters=192, kernel_size=2, padding='same', activation='relu')(conv1)
            conv2_2 = Conv1D(filters=192, kernel_size=3, padding='same', activation='relu')(conv1)
            conv2 = Concatenate()([conv2_1, conv2_2])
            conv2 = BatchNormalization()(conv2)
            
            # Third convolutional block
            conv3 = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(conv2)
            conv3 = BatchNormalization()(conv3)
            
            # Bidirectional LSTM layers
            lstm1 = Bidirectional(LSTM(256, return_sequences=True))(conv3)
            lstm1 = Dropout(0.3)(lstm1)
            
            lstm2 = Bidirectional(LSTM(192, return_sequences=False))(lstm1)
            lstm2 = Dropout(0.3)(lstm2)
            
            # Fully connected layers
            dense1 = Dense(512, activation='relu')(lstm2)
            dense1 = BatchNormalization()(dense1)
            dense1 = Dropout(0.3)(dense1)
            
            dense2 = Dense(256, activation='relu')(dense1)
            dense2 = BatchNormalization()(dense2)
            dense2 = Dropout(0.2)(dense2)
            
            outputs = Dense(self.forecast_horizon)(dense2)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Count parameters (for verification)
        total_params = model.count_params()
        print(f"Total model parameters: {total_params:,}")
        
        # Use a slower learning rate for larger models to avoid instability
        learning_rate = 0.001 if self.model_size in ['small', 'medium'] else 0.0005
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # For GPU, use AMP (Automatic Mixed Precision) when available
        try:
            # TensorFlow 2.4+
            if hasattr(tf.keras.mixed_precision, 'LossScaleOptimizer'):
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
                print("Using mixed precision optimizer")
        except Exception as e:
            print(f"Could not use mixed precision optimizer: {e}")

        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def _build_lstm_model(self, input_shape):
        """
        Build a LSTM model.
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Compiled keras model
        """
        if self.model_size == 'xlarge':
            model = Sequential()
            model.add(Bidirectional(LSTM(256, return_sequences=True, input_shape=input_shape)))
            model.add(Dropout(0.3))
            model.add(Bidirectional(LSTM(256, return_sequences=True)))
            model.add(Dropout(0.3))
            model.add(Bidirectional(LSTM(128)))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(self.forecast_horizon))
        else:
            model = Sequential()
            model.add(LSTM(192, activation='relu', input_shape=input_shape))
            model.add(Dropout(0.3))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(self.forecast_horizon))
            
        # Count parameters for verification
        total_params = model.count_params()
        print(f"Total model parameters: {total_params:,}")
            
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
        # Import tensorflow at the beginning of the method to ensure availability
        
        
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
        
        # Print shapes
        print(f"Training data shape: X = {X_train.shape}, y = {y_train.shape}")
        print(f"Validation data shape: X = {X_val.shape}, y = {y_val.shape}")
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint('best_demand_model.h5', save_best_only=True)
        ]
        
        # TensorBoard logging
        try:
            log_dir = os.path.join("logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
            os.makedirs(log_dir, exist_ok=True)
            callbacks.append(tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                profile_batch='500,520'  # Profile batch for performance analysis
            ))
        except Exception as e:
            print(f"Could not set up TensorBoard logging: {e}")
        
        # Build and train model based on type
        if self.model_type == 'cnn_lstm':
            self.model = self._build_cnn_lstm_model((X.shape[1], X.shape[2]))
        elif self.model_type == 'lstm':
            self.model = self._build_lstm_model((X.shape[1], X.shape[2]))
        
        # Print model summary
        self.model.summary()
        
        # Check if we can use GPU
        if tf.config.list_physical_devices('GPU'):
            print("GPU is available for training! ✓")
            # Set device specific batch size
            effective_batch_size = self.batch_size
        else:
            print("Training on CPU (GPU not available)")
            # Use smaller batch size for CPU
            effective_batch_size = min(32, self.batch_size)
        
        # Train model
        start_time = time.time()
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=effective_batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.1f} seconds")
        
        # Plot training history
        self._plot_history(self.history)
        
        # Print memory usage
        try:
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"Could not get memory usage: {e}")
        
        return self
    
    def _plot_history(self, history):
        """Plot training and validation loss."""
        try:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='Train MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('Model MAE')
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('demand_forecast_training_history.png')
            plt.close()
        except Exception as e:
            print(f"Could not plot training history: {e}")
    
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
        
        # Generate confidence intervals via Monte Carlo Dropout if requested
        if return_conf_int:
            # Use MC Dropout for uncertainty estimation
            n_samples = 20
            dropout_preds = []
            
            # Create a function that runs the model in training mode (with dropout active)
            def mc_predict(model, x):
                return model(x, training=True).numpy()
            
            # Get multiple predictions with dropout active
            for _ in range(n_samples):
                dropout_preds.append(mc_predict(self.model, X))
            
            dropout_preds = np.array(dropout_preds)
            mean_preds = np.mean(dropout_preds, axis=0)
            std_preds = np.std(dropout_preds, axis=0)
            
            # Inverse transform mean and std to original scale
            mean_scaled = self.scaler_y.inverse_transform(mean_preds)
            
            # Scale the standard deviation
            y_scale = self.scaler_y.scale_
            std_scaled = std_preds * y_scale[0]
            
            # 95% confidence intervals (±2σ)
            lower_bounds = np.maximum(0, mean_scaled - 2 * std_scaled)  # No negative predictions
            upper_bounds = mean_scaled + 2 * std_scaled
            
            # Use the mean as our prediction
            predictions_orig = mean_scaled[-1] if mean_scaled.shape[0] > 0 else []
            lower_bounds_last = lower_bounds[-1] if lower_bounds.shape[0] > 0 else []
            upper_bounds_last = upper_bounds[-1] if upper_bounds.shape[0] > 0 else []
        else:
            # Inverse transform predictions to original scale
            predictions_orig = self.scaler_y.inverse_transform(predictions)[-1] if predictions.shape[0] > 0 else []
        
        # Create forecast DataFrame
        try:
            last_date = pd.to_datetime(data['date'].iloc[-1])
            dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=periods_ahead,
                freq='D'
            )
            
            if len(predictions_orig) < periods_ahead:
                print(f"Warning: Predictions length ({len(predictions_orig)}) is less than forecast horizon ({periods_ahead})")
                # Pad with NaN values if needed
                padded_predictions = np.pad(predictions_orig, 
                                        (0, max(0, periods_ahead - len(predictions_orig))),
                                        'constant', 
                                        constant_values=np.nan)
                
                forecast_df = pd.DataFrame({
                    'date': dates,
                    'forecast': padded_predictions[:periods_ahead]
                })
            else:
                forecast_df = pd.DataFrame({
                    'date': dates,
                    'forecast': predictions_orig[:periods_ahead]
                })
            
            # Add product and retailer IDs if they exist in original data
            for col in ['product_id', 'retailer_id', 'entity_id']:
                if col in data.columns:
                    forecast_df[col] = data[col].iloc[-1]
            
            # Add confidence intervals if requested
            if return_conf_int:
                if len(lower_bounds_last) < periods_ahead:
                    padded_lower = np.pad(lower_bounds_last, 
                                        (0, max(0, periods_ahead - len(lower_bounds_last))),
                                        'constant', 
                                        constant_values=np.nan)
                    padded_upper = np.pad(upper_bounds_last, 
                                        (0, max(0, periods_ahead - len(upper_bounds_last))),
                                        'constant', 
                                        constant_values=np.nan)
                    forecast_df['lower_bound'] = padded_lower[:periods_ahead]
                    forecast_df['upper_bound'] = padded_upper[:periods_ahead]
                else:
                    forecast_df['lower_bound'] = lower_bounds_last[:periods_ahead]
                    forecast_df['upper_bound'] = upper_bounds_last[:periods_ahead]
            
        except Exception as e:
            print(f"Error creating forecast DataFrame: {e}")
            return pd.DataFrame(columns=['date', 'forecast'])
        
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
            'target_col': self.target_col,
            'model_size': self.model_size
        }
        
        joblib.dump(params, filepath + '_params.pkl')
        joblib.dump(self.scaler_x, filepath + '_scaler_x.pkl')
        joblib.dump(self.scaler_y, filepath + '_scaler_y.pkl')