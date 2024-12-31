from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
import os
from dotenv import load_dotenv
import json
from train_model import HybridPredictor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/predict/hybrid")
async def predict_gas_hybrid(request: Request):
    try:
        # Load the trained model
        model = HybridPredictor.load()
        
        # Load training metadata
        with open('models/training_metadata.json', 'r') as f:
            training_metadata = json.load(f)
        
        # Get current gas price (last value from training data)
        current_gas_price = float(model.last_training_data[-1])
        logger.info(f"Current gas price: {current_gas_price:.4f}")
        
        # Generate predictions
        prediction_steps = [5/60, 1, 24]  # 5min, 1hour, 24hours in hours
        predictions = model.predict(model.last_training_data, len(prediction_steps))
        
        logger.info("Predictions generated:")
        logger.info(f"5 minutes: {predictions[0]:.4f}")
        logger.info(f"1 hour: {predictions[1]:.4f}")
        logger.info(f"24 hours: {predictions[2]:.4f}")
        
        # Calculate prediction timestamps
        current_time = datetime.utcnow()
        response = {
            "current_gas_price": current_gas_price,
            "predictions": {
                "5min": {
                    "predicted_gas_price": float(predictions[0]),
                    "prediction_for": (current_time + timedelta(minutes=5)).isoformat()
                },
                "1hour": {
                    "predicted_gas_price": float(predictions[1]),
                    "prediction_for": (current_time + timedelta(hours=1)).isoformat()
                },
                "24hours": {
                    "predicted_gas_price": float(predictions[2]),
                    "prediction_for": (current_time + timedelta(hours=24)).isoformat()
                }
            },
            "model_info": {
                "last_trained": training_metadata['training_time'],
                "training_data_range": {
                    "start": training_metadata['data_start'],
                    "end": training_metadata['data_end'],
                    "samples": training_metadata['num_samples']
                }
            },
            "generated_at": current_time.isoformat(),
            "model_type": "hybrid_arima_lstm",
            "status": "success"
        }
        
        logger.info(f"Returning response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
