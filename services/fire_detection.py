"""
Fire detection service for the SeekLiyab application.

This module provides functions for fire risk assessment and classification
based on sensor readings from different areas.
"""

# Constants for fire detection thresholds
FIRE_DETECTION_THRESHOLDS = {
    "fire_detected": {
        "temperature": 50,
        "smoke": 7,
        "carbon_monoxide": 5
    },
    "potential_fire": {
        "temperature": 35,
        "smoke": 3,
        "carbon_monoxide": 2
    }
}


def classify_fire_risk(sensor_data):
    """
    Classify fire risk level based on sensor readings.
    
    Parameters:
        sensor_data (dict or pandas.Series): Sensor readings data with keys:
            - temperature_reading
            - smoke_reading
            - carbon_monoxide_reading
        
    Returns:
        str: Risk classification as "Fire Detected", "Potential Fire", or "No Fire"
    """
    thresholds = FIRE_DETECTION_THRESHOLDS
    
    # Check if we have a fire detection
    if (sensor_data['temperature_reading'] > thresholds["fire_detected"]["temperature"] and 
        sensor_data['smoke_reading'] > thresholds["fire_detected"]["smoke"] and 
        sensor_data['carbon_monoxide_reading'] > thresholds["fire_detected"]["carbon_monoxide"]):
        return "Fire Detected"
    
    # Check if we have a potential fire
    elif (sensor_data['temperature_reading'] > thresholds["potential_fire"]["temperature"] and 
          sensor_data['smoke_reading'] > thresholds["potential_fire"]["smoke"] and 
          sensor_data['carbon_monoxide_reading'] > thresholds["potential_fire"]["carbon_monoxide"]):
        return "Potential Fire"
    
    # No fire risk detected
    else:
        return "No Fire"


def should_trigger_alarm(sensor_data):
    """
    Determine if an alarm should be triggered based on sensor readings.
    
    Parameters:
        sensor_data (dict or pandas.Series): Sensor readings data
        
    Returns:
        bool: True if an alarm should be triggered, False otherwise
    """
    risk_level = classify_fire_risk(sensor_data)
    return risk_level == "Fire Detected" 