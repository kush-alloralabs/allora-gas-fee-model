from pathlib import Path
from typing import List
import os
from dotenv import load_dotenv

class Config:
    """Configuration handler for the Ethereum Gas Predictor."""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Data collection settings
        self.years_of_history = int(os.getenv('YEARS_OF_HISTORY', 2))
        self.data_interval = os.getenv('DATA_INTERVAL', '1h')
        self.prediction_intervals = os.getenv('PREDICTION_INTERVALS', '5m,1h,1d').split(',')
        
        # API settings
        self.owlracle_api_key = os.getenv('OWLRACLE_API_KEY')
        self.api_host = os.getenv('API_HOST', '0.0.0.0')
        self.api_port = int(os.getenv('API_PORT', 8000))
        
        # Model settings
        self.model_type = os.getenv('MODEL_TYPE', 'deep_ar')
        self.training_split = float(os.getenv('TRAINING_SPLIT', 0.7))
        self.validation_split = float(os.getenv('VALIDATION_SPLIT', 0.15))
        self.test_split = float(os.getenv('TEST_SPLIT', 0.15))
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        assert self.owlracle_api_key, "Owlracle API key is required"
        assert sum([self.training_split, self.validation_split, self.test_split]) == 1.0, \
            "Data splits must sum to 1.0"
        assert self.data_interval in ['5m', '1h', '1d'], \
            "Invalid data interval"
        assert all(interval in ['5m', '1h', '1d'] for interval in self.prediction_intervals), \
            "Invalid prediction interval"

config = Config() 