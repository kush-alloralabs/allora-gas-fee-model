# Allora Gas Fee Model

# Allora Gas Fee Model

## Overview
A machine learning model for predicting Ethereum gas fees using historical data from Owlracle API. The system employs a hybrid LSTM architecture with regularization and batch normalization for robust predictions.

## Dataset
- **Source**: Owlracle API
- **Timeframe**: 1-hour candles
- **Features**: 
  - Open, High, Low, Close gas prices
  - Average gas used
  - Timestamp
- **Data Range**: 1000 most recent candles
- **Training Split**: First 66% of data
- **Testing Split**: Last 33% of data

## Model Architecture
- **Type**: LSTM (Long Short-Term Memory)
- **Structure**:
  - Input Layer: Sequence length of 20
  - LSTM Layer 1: 96 units with L2 regularization
  - LSTM Layer 2: 48 units with L2 regularization
  - Dense Layers: 16 units → 1 unit
  - Batch Normalization and Dropout layers
- **Loss Function**: Huber Loss
- **Optimizer**: Adam (lr=0.002)

## Running the Pipeline

### 1. Training
```bash
python src/api/train_model.py --timeframe 1h
```
This will:
- Fetch historical data
- Train the LSTM model
- Save model and metadata to `models/` directory

### 2. Backtesting
```bash
python src/api/backtest.py --timeframe 1h
```
Generates `backtest_results.png` with three plots:
1. Actual vs Predicted prices
2. Prediction Errors
3. Direction Match Heatmap

### 3. Making Predictions
```bash
python src/api/main.py
```
Starts FastAPI server for real-time predictions.

## Interpreting Results

### Backtest Metrics
- **MAE**: Mean Absolute Error in Gwei
- **RMSE**: Root Mean Square Error in Gwei
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: Percentage of correct price movement predictions

### Visualization
1. **Top Plot**: Blue line (actual) vs Red line (predicted)
   - Close tracking indicates good model fit
   - Divergence shows prediction challenges

2. **Middle Plot**: Green line shows prediction errors
   - Oscillation around zero is normal
   - Large spikes indicate prediction challenges

3. **Bottom Plot**: Direction prediction accuracy
   - Green: Correct direction prediction
   - Red: Incorrect direction prediction

## Code References

Model Architecture:

```224:256:src/api/train_model.py
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
```


Backtest Implementation:

```12:110:src/api/backtest.py
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
```


## Future Improvements
- Feature engineering for gas price spikes
- Ensemble methods for improved stability
- Dynamic sequence length based on market volatility
- Real-time model updating
