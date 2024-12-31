import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
import os
import joblib
from dotenv import load_dotenv
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
import json
import tensorflow as tf
import argparse
from datetime import timezone
import pickle
import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Model
import ta
from fetch_data import fetch_owlracle_data
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_api_key():
    """Get Owlracle API key from environment variable"""
    api_key = os.getenv('OWLRACLE_API_KEY')
    if not api_key:
        raise ValueError("OWLRACLE_API_KEY environment variable not set")
    return api_key

def get_train_val_test_split(data, train_size=0.7, val_size=0.15):
    """Split data into training, validation and test sets based on timestamps"""
    
    # Sort by timestamp
    data = data.sort_values('timestamp')
    
    # Calculate split points using timestamps
    total_duration = data['timestamp'].max() - data['timestamp'].min()
    train_end = data['timestamp'].min() + (total_duration * train_size)
    val_end = train_end + (total_duration * val_size)
    
    # Split the data
    train_data = data[data['timestamp'] <= train_end].copy()
    val_data = data[(data['timestamp'] > train_end) & (data['timestamp'] <= val_end)].copy()
    test_data = data[data['timestamp'] > val_end].copy()
    
    # Log the splits
    logger.info(f"\nData split summary:")
    logger.info(f"Total period: {data['timestamp'].min()} to {data['timestamp'].max()}")
    logger.info(f"Training: {train_data['timestamp'].min()} to {train_data['timestamp'].max()} ({len(train_data)} samples)")
    logger.info(f"Validation: {val_data['timestamp'].min()} to {val_data['timestamp'].max()} ({len(val_data)} samples)")
    logger.info(f"Testing: {test_data['timestamp'].min()} to {test_data['timestamp'].max()} ({len(test_data)} samples)")
    
    return train_data, val_data, test_data

class HybridPredictor:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.feature_columns = ['avgGas']
        self.target_min = None
        self.target_max = None
        self.model = None
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

    def scale_target(self, x):
        """Scale values to [-1, 1] range"""
        if self.target_min is None or self.target_max is None:
            raise ValueError("Scaling parameters not set. Run prepare_data first.")
        return 2 * (x - self.target_min) / (self.target_max - self.target_min) - 1

    def inverse_scale_target(self, x):
        """Inverse scale from [-1, 1] to original range"""
        if self.target_min is None or self.target_max is None:
            raise ValueError("Scaling parameters not set. Run prepare_data first.")
        return (x + 1) * (self.target_max - self.target_min) / 2 + self.target_min

    def prepare_data(self, data):
        """Prepare data for training/prediction"""
        data = data.copy()
        
        # Debug info
        logging.info(f"Available columns in data: {data.columns.tolist()}")
        
        # We'll use close price as our target, with fallbacks
        if 'close' not in data.columns:
            raise ValueError("Required 'close' column not found in data")
        
        prices = data['close'].values
        logging.info(f"Using 'close' prices for training. Range: {prices.min():.2f} - {prices.max():.2f}")
        
        # Store scaling parameters
        self.target_min = float(prices.min())
        self.target_max = float(prices.max())
        
        logging.info(f"Setting scaling parameters - Min: {self.target_min:.2f}, Max: {self.target_max:.2f}")
        
        # Create sequences
        X = []
        y = []
        
        for i in range(len(prices) - self.sequence_length):
            X.append(prices[i:(i + self.sequence_length)])
            y.append(prices[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale both X and y
        X_scaled = self.scale_target(X)
        y_scaled = self.scale_target(y)
        
        return X_scaled.reshape(-1, self.sequence_length, 1), y_scaled
        
    def fit(self, data):
        """Train the model on the provided data"""
        try:
            # Log training timeframe
            logger.info("\nTraining Timeframe:")
            logger.info(f"Start: {data['timestamp'].min()}")
            logger.info(f"End: {data['timestamp'].max()}")
            logger.info(f"Total samples: {len(data)}")
            
            # Prepare data
            X, y = self.prepare_data(data)
            
            # Reshape X for LSTM [samples, time steps, features]
            X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Create model
            self.model = self.create_model()
            
            # Create callbacks - just use EarlyStopping for now
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-6,
                    mode='min'
                )
            ]
            
            # Train model
            logging.info("Starting model training...")
            history = self.model.fit(
                X_reshaped,
                y,
                epochs=300,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            logging.info("Model training completed")
            
            # Save the model immediately after training
            self.save()
            
            return history
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise

    def predict(self, data):
        """Make predictions using the trained model"""
        try:
            if 'close' not in data.columns:
                raise ValueError("Required 'close' column not found in data")
            
            sequence = data['close'].values[-self.sequence_length:]
            
            # # Log input range
            # logging.info("Prediction pipeline:")
            # logging.info(f"1. Input sequence range: {sequence.min():.2f} - {sequence.max():.2f}")
            # logging.info(f"2. Scaling parameters - Min: {self.target_min:.2f}, Max: {self.target_max:.2f}")
            
            # Scale input
            X_scaled = self.scale_target(sequence)
            # logging.info(f"3. Scaled features range: {X_scaled.min():.2f} - {X_scaled.max():.2f}")
            
            # Reshape for model
            X_reshaped = X_scaled.reshape((1, self.sequence_length, 1))
            
            # Make prediction
            y_pred_scaled = self.model.predict(X_reshaped, verbose=0)[0][0]
            # logging.info(f"4. Raw prediction (scaled): {y_pred_scaled:.4f}")
            
            # Inverse transform prediction
            y_pred = self.inverse_scale_target(y_pred_scaled)
            # logging.info(f"5. Final prediction: {y_pred:.2f}")
            
            return y_pred
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise

    def create_model(self):
        """Create and compile the model"""
        model = Sequential([
            Input(shape=(self.sequence_length, 1)),
            
            # First LSTM layer - keep current size but adjust regularization
            LSTM(96, return_sequences=True, 
                 kernel_regularizer=l2(0.008),  # Slightly reduced regularization
                 recurrent_regularizer=l2(0.008)),  # Added recurrent regularization
            BatchNormalization(),
            Dropout(0.15),  # Slightly reduced dropout
            
            # Second LSTM layer
            LSTM(48, 
                 kernel_regularizer=l2(0.008),
                 recurrent_regularizer=l2(0.008)),
            BatchNormalization(),
            Dropout(0.15),
            
            # Adjusted dense layers
            Dense(16, activation='relu'),  # Increased units
            BatchNormalization(),
            Dense(1, activation='tanh')
        ])
        
        # Adjusted optimizer settings
        model.compile(
            optimizer=Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999),
            loss='huber',  # Changed to Huber loss for better handling of spikes
            metrics=['mae']
        )
        
        return model

    def save(self, model_dir='models'):
        """Save the model and scaling parameters"""
        try:
            os.makedirs(model_dir, exist_ok=True)
            
            # Save Keras model
            model_path = os.path.join(model_dir, 'model.keras')
            self.model.save(model_path, save_format='keras')
            
            # Save scaling parameters
            metadata = {
                'target_min': self.target_min,
                'target_max': self.target_max,
                'sequence_length': self.sequence_length
            }
            
            metadata_path = os.path.join(model_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            logging.info(f"Model and metadata saved successfully to {model_dir}")
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load(cls, model_dir='models'):
        """Load a saved model and its parameters"""
        try:
            instance = cls()
            
            # Load model
            model_path = os.path.join(model_dir, 'model.keras')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            instance.model = load_model(model_path)
            
            # Load metadata
            metadata_path = os.path.join(model_dir, 'metadata.json')
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
                
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Set parameters
            instance.target_min = metadata['target_min']
            instance.target_max = metadata['target_max']
            instance.sequence_length = metadata['sequence_length']
            
            return instance
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

def get_training_data(df):
    """Get first 3 months of data for training"""
    # Calculate the midpoint
    midpoint = df['timestamp'].min() + (df['timestamp'].max() - df['timestamp'].min()) / 2
    
    # Get first 3 months
    train_df = df[df['timestamp'] < midpoint].copy()
    logger.info(f"Training data range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    
    return train_df[['timestamp', 'avg_gas_price']].values

def get_testing_data(df):
    """Get last 3 months of data for testing"""
    # Calculate the midpoint
    midpoint = df['timestamp'].min() + (df['timestamp'].max() - df['timestamp'].min()) / 2
    
    # Get last 3 months
    test_df = df[df['timestamp'] >= midpoint].copy()
    logger.info(f"Testing data range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    
    return test_df[['timestamp', 'avg_gas_price']].values

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeframe', default='1h', choices=['1h', '1d'])
    args = parser.parse_args()
    
    try:
        # Fetch data
        logging.info("Fetching data...")
        data = fetch_owlracle_data(timeframe=args.timeframe, candles=1000)
        filtered_data = data[333:1000]
        # Create and train model
        logging.info("Creating model...")
        model = HybridPredictor()
        
        logging.info("Training model...")
        logging.info(filtered_data.timestamp.min())
        logging.info(filtered_data.timestamp.max())
        history = model.fit(filtered_data)
        
        # Save the model
        logging.info("Saving model...")
        model.save()
        
        logging.info("Model training and saving completed successfully")
        
    except Exception as e:
        logging.error(f"Error in training process: {str(e)}")
        raise
