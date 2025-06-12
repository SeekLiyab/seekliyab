"""
Database service for the SeekLiyab fire detection system.

This module provides functions for interacting with the Supabase database,
including querying, inserting, and updating sensor readings and other data.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from st_supabase_connection import SupabaseConnection
from services.utils import format_timestamp
from services.fire_detection import predict_fire_risk


def get_supabase_connection():
    """
    Get a connection to the Supabase database.
    
    Returns:
        SupabaseConnection: A connection to the Supabase database
    """
    return st.connection("supabase", type=SupabaseConnection)


def get_recent_readings_for_area(area_name, limit=20):
    """
    Get recent sensor readings for a specific area.
    
    Parameters:
        area_name (str): The name of the area to get readings for
        limit (int, optional): Maximum number of readings to retrieve
        
    Returns:
        pandas.DataFrame or None: DataFrame with sensor readings or None if no data
    """
    conn = get_supabase_connection()
    
    # Query the latest data
    rows = conn.table("sensor_readings") \
            .select("timestamp, area_name, temperature_reading, air_quality_reading, carbon_monoxide_reading, smoke_reading") \
            .eq("area_name", area_name) \
            .order("timestamp", desc=True) \
            .limit(limit) \
            .execute()
    
    # Convert to DataFrame
    if rows.data and len(rows.data) > 0:
        df = pd.DataFrame(rows.data)
        
        # Apply fire risk classification
        df['fire_risk'] = df.apply(classify_fire_risk, axis=1)
        df['timestamp'] = df['timestamp'].apply(format_timestamp)
        
        return df
    
    return None


def get_readings_for_timeframe(area_name, hours=24):
    """
    Get sensor readings for a specific area within a timeframe.
    
    Parameters:
        area_name (str): The name of the area to get readings for
        hours (int, optional): Number of hours to look back
        
    Returns:
        pandas.DataFrame or None: DataFrame with sensor readings or None if no data
    """
    conn = get_supabase_connection()
    
    # Calculate the timestamp for the start of the timeframe
    start_time = (datetime.now() - timedelta(hours=hours)).isoformat()
    
    # Query the data within the timeframe
    rows = conn.table("sensor_readings") \
            .select("timestamp, area_name, temperature_reading, air_quality_reading, carbon_monoxide_reading, smoke_reading") \
            .eq("area_name", area_name) \
            .gte("timestamp", start_time) \
            .order("timestamp") \
            .execute()
    
    # Convert to DataFrame
    if rows.data and len(rows.data) > 0:
        df = pd.DataFrame(rows.data)
        
        # Apply fire risk classification
        df['fire_risk'] = df.apply(classify_fire_risk, axis=1)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    return None 


def classify_fire_risk(row):
    """Classify fire risk for a row of sensor data using the updated ML model."""
    result = predict_fire_risk(row.to_dict())
    return result['prediction'] 