import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_model import BaseModel

class MovingAverageModel(BaseModel):
    """Simple moving average model for testing the pipeline."""
    
    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        self.historical_data = None
        
    def train(self, data: pd.DataFrame) -> None:
        """Store historical data for making predictions."""
        self.historical_data = data.copy()
        self.historical_data = self.historical_data.sort_values('timestamp')
    
    def predict(self, interval: str) -> Dict[str, Any]:
        """Make predictions using simple moving average."""
        if self.historical_data is None:
            raise ValueError("Model must be trained before making predictions")
            
        # Calculate moving average of last window_size entries
        last_prices = self.historical_data['gas_price'].tail(self.window_size)
        prediction = last_prices.mean()
        
        # Calculate confidence intervals using standard deviation
        std_dev = last_prices.std()
        
        return {
            'prediction': prediction,
            'confidence_interval_lower': prediction - 2 * std_dev,
            'confidence_interval_upper': prediction + 2 * std_dev,
            'interval': interval
        }
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: DataFrame containing test data
            
        Returns:
            Dictionary containing evaluation metrics (MAE and RMSE)
        """
        if self.historical_data is None:
            raise ValueError("Model must be trained before evaluation")
            
        # Make predictions for each point in test data
        predictions = []
        actuals = test_data['gas_price'].values
        
        for i in range(len(test_data)):
            # Use all available data up to this point for prediction
            train_data = pd.concat([
                self.historical_data,
                test_data.iloc[:i]
            ]).sort_values('timestamp')
            
            last_prices = train_data['gas_price'].tail(self.window_size)
            predictions.append(last_prices.mean())
        
        # Calculate metrics
        predictions = np.array(predictions)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        return {
            'mae': mae,
            'rmse': rmse
        }