from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class BaseModel(ABC):
    """Base class for all gas price prediction models."""
    
    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the model on historical data.
        
        Args:
            data: DataFrame containing historical gas price data
        """
        pass
    
    @abstractmethod
    def predict(self, interval: str) -> Dict[str, Any]:
        """
        Make gas price predictions for the specified interval.
        
        Args:
            interval: Prediction interval (5m, 1h, 1d)
            
        Returns:
            Dictionary containing predictions and confidence intervals
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: DataFrame containing test data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass 