"""
Utility functions for the SeekLiyab fire detection system.

This module provides common utility functions used across the application
for data processing, formatting, and system operations.
"""

import pandas as pd
from datetime import datetime


def format_timestamp(timestamp_str, format_str='%b %d, %Y %I:%M:%S %p'):
    """
    Format a timestamp string to a human-readable format.
    
    Parameters:
        timestamp_str (str): ISO format timestamp string
        format_str (str): Output datetime format string
        
    Returns:
        str: Formatted timestamp
    """
    if isinstance(timestamp_str, str):
        dt = pd.to_datetime(timestamp_str)
        return dt.strftime(format_str)
    
    return timestamp_str


def get_risk_level_style(risk_level):
    """
    Get UI style attributes for a given risk level.
    
    Parameters:
        risk_level (str): The risk level ('Fire Detected', 'Potential Fire', or 'No Fire')
        
    Returns:
        dict: Dictionary with 'color' and 'icon' keys for styling
    """
    if risk_level == "Fire Detected":
        return {
            "color": "red",
            "icon": "🔥"
        }
    elif risk_level == "Potential Fire":
        return {
            "color": "orange",
            "icon": "⚠️"
        }
    else:
        return {
            "color": "green",
            "icon": "✅"
        } 