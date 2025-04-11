from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf

class ForecastEnsemble:
    """Ensemble of forecasting models for improved accuracy"""
    
    def __init__(self):
        self.models = []
        self.weights = []
    
    def add_model(self, model, weight=1.0):
        """Add a model to the ensemble"""
        self.models.append(model)
        self.weights.append(weight)
    
    def normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total = sum(self.weights)
        if total > 0:
            self.weights = [w/total for w in self.weights]
    
    def predict(self, X):
        """Make weighted ensemble prediction"""
        if not self.models:
            raise ValueError("No models in ensemble")
            
        self.normalize_weights()
        predictions = np.zeros((X.shape[0],))
        
        for i, model in enumerate(self.models):
            pred = model.predict(X, verbose=0).flatten()
            predictions += pred * self.weights[i]
            
        return predictions