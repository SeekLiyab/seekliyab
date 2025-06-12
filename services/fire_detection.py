"""
Fire detection service for the SeekLiyab application.

This module provides ML-powered fire risk assessment and classification
based on sensor readings using the trained Random Forest model.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the model path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(root_dir, "model-development", "models", "seekliyab_model.pkl")

# Fallback thresholds for backup safety checks
FIRE_DETECTION_THRESHOLDS = {
    "fire_detected": {
        "temperature_reading": 50,
        "smoke_reading": 500,
        "carbon_monoxide_reading": 600
    },
    "potential_fire": {
        "temperature_reading": 40,
        "smoke_reading": 300,
        "carbon_monoxide_reading": 450
    }
}


# Dummy class to handle pickle loading issues
class SeekLiyabFireDetector:
    """Dummy class for pickle compatibility"""
    pass


# Use cache_resource for model object
@st.cache_resource
def get_fire_detection_model():
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            return None
        logger.info(f"Loading model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# Update load_fire_detection_model to use the cached resource
@st.cache_data
def load_fire_detection_model(MODEL_PATH):
    return get_fire_detection_model()

@st.cache_data
def engineer_features(X):
  """Create engineered features using only training statistics"""

  X_engineered = X.copy()
  stats = {'temp_mean': np.float64(42.539846743295016),
    'temp_std': 18.25924899622683,
    'temp_q25': np.float64(26.81),
    'temp_q75': np.float64(62.13),
    'air_mean': np.float64(493.4169859514687),
    'air_std': 364.32276066001566,
    'co_mean': np.float64(460.31864623243933),
    'co_std': 255.01250235031887,
    'smoke_mean': np.float64(444.19220945083015),
    'smoke_std': 399.0431247723564
  }

  # Map column names to standard names for feature engineering
  temp_col = 'temperature_reading' if 'temperature_reading' in X.columns else 'temperature'
  air_col = 'air_quality_reading' if 'air_quality_reading' in X.columns else 'air_quality'
  co_col = 'carbon_monoxide_reading' if 'carbon_monoxide_reading' in X.columns else 'carbon_monoxide'
  smoke_col = 'smoke_reading' if 'smoke_reading' in X.columns else 'gas_and_smoke'

  # Temperature-based features (critical for fire detection)
  X_engineered['temp_zscore'] = (X[temp_col] - stats['temp_mean']) / stats['temp_std']
  X_engineered['temp_critical'] = (X[temp_col] > 50).astype(int)
  X_engineered['temp_extreme'] = (X[temp_col] > 60).astype(int)

  # Gas concentration features
  X_engineered['co_zscore'] = (X[co_col] - stats['co_mean']) / stats['co_std']
  X_engineered['smoke_zscore'] = (X[smoke_col] - stats['smoke_mean']) / stats['smoke_std']
  X_engineered['air_zscore'] = (X[air_col] - stats['air_mean']) / stats['air_std']

  # Interaction features (physics-based)
  X_engineered['temp_co_interaction'] = X[temp_col] * X[co_col] / 1000
  X_engineered['temp_smoke_interaction'] = X[temp_col] * X[smoke_col] / 1000
  X_engineered['co_smoke_ratio'] = X[co_col] / (X[smoke_col] + 1)

  # Fire risk composite score
  temp_risk = np.clip((X[temp_col] - 20) / 50, 0, 1)
  co_risk = np.clip((X[co_col] - 300) / 500, 0, 1)
  smoke_risk = np.clip((X[smoke_col] - 200) / 400, 0, 1)
  air_risk = np.clip((X[air_col] - 300) / 500, 0, 1)

  X_engineered['fire_risk_composite'] = (
      0.4 * temp_risk +
      0.25 * co_risk +
      0.25 * smoke_risk +
      0.1 * air_risk
  )

  # Anomaly indicators
  X_engineered['temp_anomaly'] = (abs(X_engineered['temp_zscore']) > 2).astype(int)
  X_engineered['co_anomaly'] = (abs(X_engineered['co_zscore']) > 2).astype(int)
  X_engineered['smoke_anomaly'] = (abs(X_engineered['smoke_zscore']) > 2).astype(int)
  X_engineered['total_anomalies'] = (X_engineered['temp_anomaly'] +
                                    X_engineered['co_anomaly'] +
                                    X_engineered['smoke_anomaly'])

  return X_engineered


def physics_validation(sensor_data):
    """Physics-based validation using fire science principles"""
    indicators = 0
    temp = sensor_data.get('temperature_reading', 0)
    co = sensor_data.get('carbon_monoxide_reading', 0)
    smoke = sensor_data.get('smoke_reading', 0)
    
    if temp > 50:  # Fire temperature threshold
        indicators += 1
    if co > 600:  # Fire CO threshold  
        indicators += 1
    if smoke > 500:  # Fire smoke threshold
        indicators += 1
        
    return indicators


def fallback_classification(sensor_data):
    """Fallback threshold-based classification"""
    logger.info("Using fallback threshold-based classification")
    
    thresholds = FIRE_DETECTION_THRESHOLDS
    temp = sensor_data.get('temperature_reading', 0)
    smoke = sensor_data.get('smoke_reading', 0)
    co = sensor_data.get('carbon_monoxide_reading', 0)
    
    # Check for fire detection
    if (temp > thresholds["fire_detected"]["temperature_reading"] and 
        smoke > thresholds["fire_detected"]["smoke_reading"] and 
        co > thresholds["fire_detected"]["carbon_monoxide_reading"]):
        prediction = "Fire Detected"
        risk_level = "CRITICAL"
        action = "IMMEDIATE EVACUATION"
        fire_prob = 0.9
        
    # Check for potential fire
    elif (temp > thresholds["potential_fire"]["temperature_reading"] and 
          smoke > thresholds["potential_fire"]["smoke_reading"] and 
          co > thresholds["potential_fire"]["carbon_monoxide_reading"]):
        prediction = "Potential Fire"
        risk_level = "HIGH"
        action = "INVESTIGATE IMMEDIATELY"
        fire_prob = 0.6
        
    else:
        prediction = "No Fire"
        risk_level = "LOW"
        action = "ROUTINE MONITORING"
        fire_prob = 0.1
    
    return {
        'prediction': prediction,
        'ml_prediction': prediction,
        'probabilities': {
            'Fire': fire_prob if prediction == "Fire Detected" else 0.1,
            'Potential Fire': 0.7 if prediction == "Potential Fire" else 0.2,
            'No Fire': 0.8 if prediction == "No Fire" else 0.1
        },
        'risk_level': risk_level,
        'recommended_action': action,
        'fire_probability': fire_prob,
        'potential_fire_probability': 0.7 if prediction == "Potential Fire" else 0.2,
        'alert_required': prediction in ["Fire Detected", "Potential Fire"],
        'physics_indicators': physics_validation(sensor_data),
        'confidence': 0.85,
        'sensor_data': sensor_data,
        'model_used': False
    }


def get_label(predicted_class_id):
    """Get the label from the predicted class id"""
    if predicted_class_id == 0:
        return "Fire"
    elif predicted_class_id == 1:
        return "Potential Fire"
    else:
        return "No Fire"

def predict_fire_risk(sensor_data):
    """
    Predict fire risk using the trained ML model.
    
    Parameters:
        sensor_data (dict): Sensor readings with keys:
            - temperature_reading
            - smoke_reading
            - carbon_monoxide_reading
            - air_quality_reading (optional)
            
    Returns:
        dict: Comprehensive fire risk assessment
    """
    # Load model components
    model = load_fire_detection_model( MODEL_PATH)
    
    if model is None:
        logger.warning("Model not loaded, using fallback method")
        return fallback_classification(sensor_data)
    
    else:
        logger.info("Model loaded successfully")
    
    try:
        # Engineer features
        X_engineered = engineer_features(pd.DataFrame([sensor_data]))
        print(X_engineered)

        if X_engineered is None:
            return fallback_classification(sensor_data)
        else:
            logger.info(f"Engineered features: {X_engineered}")
    
        
        # Model prediction
        if hasattr(model, 'feature_names_in_'):
            X_engineered = X_engineered[model.feature_names_in_]
        else:
            print('Model does not have feature_names_in_. Columns in X_engineered:', X_engineered.columns.tolist())
        probabilities = model.predict_proba(X_engineered)
        logger.info(f"Probabilities: {probabilities}")
        predicted_class_id = np.argmax(probabilities)
        predicted_class = get_label(predicted_class_id)
        
        return {
            'prediction': predicted_class,
            'probabilities': probabilities,
            'confidence': probabilities[0][predicted_class_id],
            'sensor_data': sensor_data,
            'model_used': True
        }
    except Exception as e:
        logger.error(f"Error predicting fire risk: {e}")
        return fallback_classification(sensor_data)



# Example usage and testing function
def test_fire_detection():
    """Test the fire detection system with sample data"""
    
    test_scenarios = [
        {
            'name': 'Normal Conditions',
            'data': {
                'temperature_reading': 22.0,
                'smoke_reading': 190,
                'carbon_monoxide_reading': 320,
                'air_quality_reading': 280
            }
        },
        {
            'name': 'Potential Fire',
            'data': {
                'temperature_reading': 45.0,
                'smoke_reading': 350,
                'carbon_monoxide_reading': 480,
                'air_quality_reading': 420
            }
        },
        {
            'name': 'Fire Detected',
            'data': {
                'temperature_reading': 65.0,
                'smoke_reading': 720,
                'carbon_monoxide_reading': 780,
                'air_quality_reading': 850
            }
        }
    ]
    
    print("🔥 Testing SeekLiyab Fire Detection System")
    print("=" * 50)
    
    for scenario in test_scenarios:
        print(f"\n📍 Scenario: {scenario['name']}")
        result = predict_fire_risk(scenario['data'])
        


def send_sms_alert(numbers, message):
    """Placeholder for sending SMS to a list of numbers."""
    # Integrate with your SMS provider here (e.g., Twilio)
    for number in numbers:
        print(f"Sending SMS to {number}: {message}")
    # You can replace this with actual SMS sending logic


# Batch prediction for a DataFrame of sensor readings
def batch_predict_fire_risk(df):
    model = get_fire_detection_model()
    if model is None or df is None or df.empty:
        return [fallback_classification(row) for _, row in df.iterrows()]
    # Engineer features for all rows at once
    X_engineered = engineer_features(df)
    if hasattr(model, 'feature_names_in_'):
        X_engineered = X_engineered[model.feature_names_in_]
    # Predict probabilities for all rows
    probabilities = model.predict_proba(X_engineered)
    preds = np.argmax(probabilities, axis=1)
    labels = [get_label(idx) for idx in preds]
    results = []
    for i, row in enumerate(df.itertuples(index=False)):
        results.append({
            'prediction': labels[i],
            'probabilities': probabilities[i],
            'confidence': probabilities[i][preds[i]],
            'sensor_data': row._asdict(),
            'model_used': True
        })
    return results

# Optimize check_and_alert_consecutive_fires to use batch prediction
def check_and_alert_consecutive_fires(area_name, n=10):
    from services.database import get_recent_readings_for_area
    df = get_recent_readings_for_area(area_name, limit=n)
    if df is None or df.empty or len(df) < n:
        return False
    df = df.sort_values('timestamp', ascending=False)
    # Use batch prediction for all rows
    results = batch_predict_fire_risk(df)
    fire_predictions = [r['prediction'] for r in results]
    if all(pred == 'Fire Detected' for pred in fire_predictions):
        numbers = st.secrets.emergency_contacts.emergency_numbers
        message = f"ALERT: {n} consecutive fire detections in {area_name}! Immediate action required."
        send_sms_alert(numbers, message)
        return True
    return False


if __name__ == "__main__":
    print(joblib.load(MODEL_PATH))
    test_fire_detection() 