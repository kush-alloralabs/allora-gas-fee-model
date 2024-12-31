import logging
import pandas as pd
import numpy as np
from train_model import HybridPredictor
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from fetch_data import fetch_owlracle_data

logger = logging.getLogger(__name__)

def backtest_model(timeframe='1h'):
    """Backtest the model with enhanced visualization"""
    try:
        # Load the trained model
        model = HybridPredictor.load()
        logger.info("Loaded trained model")
        
        # Load cached data
        full_data = pd.read_pickle('data/historical_data.pkl')
        logger.info("\nBacktesting Timeframe:")
        logger.info(f"Full data range: {full_data['timestamp'].min()} to {full_data['timestamp'].max()}")
        
        # Calculate midpoint for testing data
        midpoint = full_data['timestamp'].min() + (full_data['timestamp'].max() - full_data['timestamp'].min()) / 1.5
        test_data = full_data[full_data['timestamp'] >= midpoint].copy()
        logger.info(f"Testing period: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")
        logger.info(f"Testing on {len(test_data)} data points")
        
        # Initialize results storage
        predictions = []
        actuals = []
        timestamps = []
        
        # Sliding window prediction
        window_size = 20
        for i in range(window_size, len(test_data)):
            # Create window DataFrame
            window_df = test_data.iloc[i-window_size:i][['timestamp', 'close']].copy()
            actual_price = test_data.iloc[i]['close']
            
            try:
                pred = model.predict(window_df)
                predictions.append(pred)
                actuals.append(actual_price)
                timestamps.append(test_data.iloc[i]['timestamp'])
            except Exception as e:
                logger.error(f"Error making prediction at index {i}: {str(e)}")
                continue
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        timestamps = np.array(timestamps)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Calculate directional accuracy
        actual_direction = np.diff(actuals) > 0
        pred_direction = np.diff(predictions) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # Log metrics
        logger.info(f"\nBacktest metrics:")
        logger.info(f"MAE: {mae:.2f}")
        logger.info(f"RMSE: {rmse:.2f}")
        logger.info(f"MAPE: {mape:.2f}%")
        logger.info(f"Directional Accuracy: {directional_accuracy:.2f}%")
        
        # Create visualization
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Actual vs Predicted
        plt.subplot(311)
        plt.plot(timestamps, actuals, label='Actual', color='blue')
        plt.plot(timestamps, predictions, label='Predicted', color='red')
        plt.title('Gas Price: Actual vs Predicted')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Prediction Errors
        plt.subplot(312)
        plt.plot(timestamps, predictions - actuals, color='green')
        plt.title('Prediction Errors Over Time')
        plt.grid(True)
        
        # Plot 3: Direction Match Heatmap
        plt.subplot(313)
        direction_matches = (actual_direction == pred_direction).astype(int)
        plt.imshow([direction_matches], aspect='auto', cmap='RdYlGn')
        plt.title('Direction Match Heatmap')
        plt.colorbar(label='Direction Match (Green=Correct, Red=Incorrect)')
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.close()
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }, {
            'timestamps': timestamps,
            'actuals': actuals,
            'predictions': predictions
        }
        
    except Exception as e:
        logger.error(f"Error in backtesting: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeframe', default='1h', choices=['1h', '1d'])
    args = parser.parse_args()
    
    metrics, results = backtest_model(timeframe=args.timeframe)