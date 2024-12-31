import requests
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional
from src.config import config

class OwlracleCollector:
    """Collector for Ethereum gas price data from Owlracle."""
    
    BASE_URL = "https://api.owlracle.info/v4/eth/history"
    
    def __init__(self):
        self.api_key = config.owlracle_api_key
    
    def _get_timeframe(self, interval: str) -> str:
        """Convert interval to Owlracle API timeframe format."""
        interval_map = {
            '5m': '10m',  # Minimum is 10m for history endpoint
            '1h': '1h',
            '1d': '1d'
        }
        return interval_map.get(interval, '1h')
    
    def collect_data(self, interval: str = None) -> pd.DataFrame:
        """
        Collect gas price data from Owlracle.
        
        Args:
            interval: Data interval (5m, 1h, 1d). If None, uses config default.
            
        Returns:
            DataFrame with timestamp and gas price data.
        """
        interval = interval or config.data_interval
        timeframe = self._get_timeframe(interval)
        
        params = {
            'apikey': self.api_key,
            'timeframe': timeframe,
            'candles': 1000,  # Maximum samples
            'txfee': 'true'   # Include transaction fee data
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, list):
                raise Exception(f"Unexpected API response format: {data}")
            
            # Process the response into a DataFrame
            records = []
            for entry in data:
                try:
                    records.append({
                        'timestamp': pd.to_datetime(entry['timestamp'] * 1000, unit='ms'),
                        'gas_price': float(entry['avgGas'])  # Using average gas price
                    })
                except (KeyError, TypeError) as e:
                    print(f"Warning: Skipping malformed entry: {entry}")
                    continue
            
            if not records:
                raise Exception("No valid data points found in API response")
            
            df = pd.DataFrame(records)
            df = df.sort_values('timestamp')
            
            # If interval is 5m, we'll resample to get the desired granularity
            if interval == '5m':
                df = df.set_index('timestamp')
                df = df.resample('5T').mean().dropna()
                df = df.reset_index()
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to collect data from Owlracle: {str(e)}")