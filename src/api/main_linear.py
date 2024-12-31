# from fastapi import FastAPI, HTTPException, Request
# from pydantic import BaseModel
# from typing import List, Dict
# import pandas as pd
# from datetime import datetime, timedelta
# import numpy as np
# from sklearn.linear_model import LinearRegression
# import json
# import logging
# import requests
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# app = FastAPI()

# async def fetch_historical_data():
#     """Fetch historical gas price data from Owlracle API"""
#     try:
#         api_key = os.environ.get('OWLRACLE_API_KEY')
#         if not api_key:
#             raise ValueError("OWLRACLE_API_KEY environment variable not set")
            
#         logger.info(f"Using API key: {api_key[:5]}...")
#         url = f"https://api.owlracle.info/v4/eth/history?apikey={api_key}&candles=1000&timeframe=1h"
        
#         response = requests.get(url)
#         response.raise_for_status()
#         data = response.json()
        
#         logger.info("Successfully fetched historical data from Owlracle")
#         return data['candles'] if 'candles' in data else data  # Extract candles array
        
#     except Exception as e:
#         logger.error(f"Error fetching historical data: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to fetch historical data: {str(e)}")

# @app.get("/predict")
# @app.post("/predict")
# async def predict_gas(request: Request):
#     try:
#         # Fetch historical data
#         historical_data = await fetch_historical_data()
#         logger.info(f"Processing {len(historical_data)} historical data points")
        
#         # Extract gas price data
#         processed_data = []
#         for entry in historical_data:
#             processed_data.append({
#                 'timestamp': entry['timestamp'],
#                 'high': entry['gasPrice']['high'],
#                 'low': entry['gasPrice']['low'],
#                 'open': entry['gasPrice']['open'],
#                 'close': entry['gasPrice']['close'],
#                 'avgGas': entry['avgGas']
#             })
        
#         # Convert to DataFrame
#         df = pd.DataFrame(processed_data)
#         logger.info(f"DataFrame columns: {df.columns.tolist()}")
#         logger.info(f"DataFrame shape: {df.shape}")
        
#         # Calculate average gas price
#         df['avg_gas_price'] = (df['high'] + df['low']) / 2
#         logger.info(f"Average gas price range: {df['avg_gas_price'].min():.2f} - {df['avg_gas_price'].max():.2f}")
        
#         # Convert timestamp to datetime
#         df['timestamp'] = pd.to_datetime(df['timestamp'])
#         logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
#         # Sort by timestamp
#         df = df.sort_values('timestamp')
        
#         # Create features
#         df['hours'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600
#         logger.info(f"Hours range: {df['hours'].min():.2f} - {df['hours'].max():.2f}")
        
#         # Prepare data for model
#         X = df['hours'].values.reshape(-1, 1)
#         y = df['avg_gas_price'].values
        
#         # Fit linear regression model
#         model = LinearRegression()
#         model.fit(X, y)
#         logger.info(f"Model coefficients: {model.coef_[0]:.4f}, intercept: {model.intercept_:.4f}")
        
#         # Predict for different time horizons
#         current_hour = df['hours'].max()
#         predictions = {
#             "5min": model.predict([[current_hour + (5/60)]])[0],
#             "1hour": model.predict([[current_hour + 1]])[0],
#             "24hours": model.predict([[current_hour + 24]])[0]
#         }
        
#         logger.info("Predictions generated:")
#         logger.info(f"5 minutes: {predictions['5min']:.4f}")
#         logger.info(f"1 hour: {predictions['1hour']:.4f}")
#         logger.info(f"24 hours: {predictions['24hours']:.4f}")
        
#         # Calculate prediction timestamps
#         current_time = datetime.utcnow()
#         response = {
#             "predictions": {
#                 "5min": {
#                     "predicted_gas_price": float(predictions['5min']),
#                     "prediction_for": (current_time + timedelta(minutes=5)).isoformat()
#                 },
#                 "1hour": {
#                     "predicted_gas_price": float(predictions['1hour']),
#                     "prediction_for": (current_time + timedelta(hours=1)).isoformat()
#                 },
#                 "24hours": {
#                     "predicted_gas_price": float(predictions['24hours']),
#                     "prediction_for": (current_time + timedelta(hours=24)).isoformat()
#                 }
#             },
#             "current_gas_price": float(df['avg_gas_price'].iloc[-1]),
#             "generated_at": current_time.isoformat(),
#             "status": "success"
#         }
        
#         logger.info(f"Returning response: {response}")
#         return response
        
#     except Exception as e:
#         logger.error(f"Error in prediction pipeline: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {"message": "Gas Price Prediction API"} 