import os
import logging
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
def fetch_owlracle_data(timeframe='1h', candles=1000):
    """Fetch historical gas price data from Owlracle API"""
    def parse_date(date_str):
        try:
            return pd.to_datetime(date_str, format='ISO8601')
        except:
            return pd.to_datetime(date_str)
            
    def parse_gas_price(gas_price):
        """Parse gasPrice whether it's a string or dict"""
        if isinstance(gas_price, dict):
            return gas_price
        if isinstance(gas_price, str):
            return eval(gas_price)
        logger.error(f"Unexpected gasPrice type: {type(gas_price)}")
        return None


    # Load cached data if available
    cache_file = f'gas_data_{timeframe}.csv'
    pkl_file = 'data/historical_data.pkl'
    
    if os.path.exists(cache_file):
        logging.info("Loading data from cache")
        data = pd.read_csv(cache_file, parse_dates=['timestamp'], date_parser=parse_date)
        
        # Parse the gasPrice column if it exists
        if 'gasPrice' in data.columns:
            data['gasPrice'] = data['gasPrice'].apply(parse_gas_price)
            data['open'] = data['gasPrice'].apply(lambda x: x['open'])
            data['close'] = data['gasPrice'].apply(lambda x: x['close'])
            data['high'] = data['gasPrice'].apply(lambda x: x['high'])
            data['low'] = data['gasPrice'].apply(lambda x: x['low'])
            data = data.drop('gasPrice', axis=1)
            
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Save processed data to pickle
            data.to_pickle(pkl_file)
            logger.info(f"Saved processed data to {pkl_file}")
            
        return data
    
    try:
        # api_key = os.getenv('OWLRACLE_API_KEY')
        api_key = '481ce83db03c4662918df91a303e2fe9'
        logger.info(f"API key: {api_key}")
        if not api_key:
            raise ValueError("OWLRACLE_API_KEY environment variable not set")
            
        logger.info(f"Using API key: {api_key[:5]}...")
        url = f"https://api.owlracle.info/v4/eth/history"
        
        params = {
            'apikey': api_key,
            'candles': candles,
            'timeframe': timeframe
        }
        
        logger.info(f"Requesting {candles} candles with timeframe {timeframe}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Log the raw response details
        logger.info(f"API Response Status: {response.status_code}")
        if isinstance(data, dict):
            logger.info(f"Response keys: {data.keys()}")
            if 'candles' in data:
                logger.info(f"Received {len(data['candles'])} candles")
            if 'message' in data:
                logger.info(f"API Message: {data['message']}")
        
        # Extract candles array if present
        candles_data = data['candles'] if 'candles' in data else data
        
        # Convert to DataFrame
        df = pd.DataFrame(candles_data)
        logger.info(f"Created DataFrame with {len(df)} rows")
        print(df.head())
        
        # Parse the gasPrice column
        if 'gasPrice' in df.columns:
            df['gasPrice'] = df['gasPrice'].apply(parse_gas_price)
            df['open'] = df['gasPrice'].apply(lambda x: x['open'])
            df['close'] = df['gasPrice'].apply(lambda x: x['close'])
            df['high'] = df['gasPrice'].apply(lambda x: x['high'])
            df['low'] = df['gasPrice'].apply(lambda x: x['low'])
            df = df.drop('gasPrice', axis=1)
        
        # Handle timestamp conversion
        if 'timestamp' in df.columns:
            # Try different timestamp formats
            try:
                # If timestamp is already in seconds
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            except ValueError:
                try:
                    # If timestamp is ISO format
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except ValueError:
                    # If timestamp is in milliseconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Save to cache
        df.to_csv(cache_file, index=False)
        logger.info(f"Saved {len(df)} records to cache")
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save processed data to pickle
        df.to_pickle(pkl_file)
        logger.info(f"Saved processed data to {pkl_file}")
        
        logger.info("Successfully fetched historical data from Owlracle")
        logger.info(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        if hasattr(e.response, 'status_code'):
            logger.error(f"Status code: {e.response.status_code}")
            if e.response.status_code == 403:
                logger.error("Authentication failed. Please check your API key.")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return None

def fetch_historical_data(timeframe='1h'):
    """Fetch historical gas price data"""
    try:
        # Load from cache if available
        cache_file = f'data/historical_data.pkl'
        if os.path.exists(cache_file):
            logger.info("Loading data from cache")
            # Read CSV and parse gasPrice column
            data = pd.read_csv('src/api/data/owlracle_1h_1000.csv')
            
            # Parse the gasPrice string into separate columns
            data['gasPrice'] = data['gasPrice'].apply(eval)  # Convert string to dict
            data['open'] = data['gasPrice'].apply(lambda x: x['open'])
            data['close'] = data['gasPrice'].apply(lambda x: x['close'])
            data['high'] = data['gasPrice'].apply(lambda x: x['high'])
            data['low'] = data['gasPrice'].apply(lambda x: x['low'])
            
            # Drop the original gasPrice column
            data = data.drop('gasPrice', axis=1)
            
            logger.info(f"Available columns: {data.columns.tolist()}")
            
            # Save processed data
            data.to_pickle(cache_file)
            return data

        logger.info("Fetching fresh data...")
        # ... rest of the fetch logic ...
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        data = fetch_owlracle_data()
        if data is not None:
            print(f"Fetched {len(data)} records")
            print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
            print(f"Columns: {data.columns.tolist()}")
            print("\nSample data:")
            print(data.head())
        else:
            print("Failed to fetch data")
    except Exception as e:
        print(f"Error: {str(e)}")