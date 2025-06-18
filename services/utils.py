"""
Utility functions for the SeekLiyab fire detection system.

This module provides common utility functions used across the application
for data processing, formatting, and system operations.
"""

import pandas as pd
from datetime import datetime


def format_timestamp(timestamp_str, format_str='%Y-%m-%d %H:%M:%S PHT'):
    """
    Format a timestamp string to a human-readable format in Philippine Time.
    
    Parameters:
        timestamp_str (str): ISO format timestamp string
        format_str (str): Output datetime format string
        
    Returns:
        str: Formatted timestamp in Philippine Time
    """
    if isinstance(timestamp_str, str):
        try:
            # Parse the timestamp and convert to Philippine Time
            dt = pd.to_datetime(timestamp_str)
            # Assume UTC if no timezone info, then convert to Philippine Time
            if dt.tz is None:
                dt = dt.tz_localize('UTC')
            dt = dt.tz_convert('Asia/Manila')
            return dt.strftime(format_str)
        except Exception:
            # If parsing fails, return original
            return timestamp_str
    
    return timestamp_str


def get_risk_level_style(risk_level):
    """
    Get UI style attributes for a given risk level.
    
    Parameters:
        risk_level (str): The risk level from various sources:
            - CSV data: 'LOW', 'HIGH', 'CRITICAL'  
            - ML predictions: 'Fire Detected', 'Fire', 'Potential Fire', 'No Fire'
        
    Returns:
        dict: Dictionary with 'color' and 'icon' keys for styling
    """
    # Handle CSV data risk levels
    if risk_level == "CRITICAL":
        return {
            "color": "#8B0000",  # Dark red for critical
            "icon": "🔥"
        }
    elif risk_level == "HIGH":
        return {
            "color": "#FF8C00",  # Dark orange for high risk
            "icon": "⚠️"
        }
    elif risk_level == "LOW":
        return {
            "color": "#32CD32",  # Lime green for low risk
            "icon": "✅"
        }
    # Handle ML prediction values (backward compatibility)
    elif risk_level == "Fire Detected" or risk_level == "Fire":
        return {
            "color": "#8B0000",  # Dark red
            "icon": "🔥"
        }
    elif risk_level == "Potential Fire":
        return {
            "color": "#FF8C00",  # Dark orange
            "icon": "⚠️"
        }
    elif risk_level == "No Fire":
        return {
            "color": "#32CD32",  # Lime green
            "icon": "✅"
        }
    else:
        # Default fallback for unknown values
        return {
            "color": "#808080",  # Gray for unknown
            "icon": "❓"
        } 