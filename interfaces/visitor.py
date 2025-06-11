"""
Visitor interface for the SeekLiyab fire detection system.

This module provides a visual interface for visitors to select different
areas of the building to view their current fire detection status.
"""

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from components.footer import display_footer


def create_area_selection(area_name, column):
    """
    Create an interactive area selection container.
    
    Parameters:
        area_name (str): The name of the area to create selection for
        column (streamlit.delta_generator.DeltaGenerator): The column to render in
        
    Returns:
        None: The component is rendered directly to the Streamlit interface
    """
    with column.container(border=True):
        area_click = streamlit_image_coordinates(
            "app/static/images/seekliyab-banner-f.png", 
            key=f"area{area_name.split()[-1]}", 
            use_column_width=True
        )
        
        if area_click:
            # Store the area in session state before switching pages
            st.session_state.selected_area = area_name
            # Update query params
            st.query_params["area"] = area_name
            # Switch to dashboard page
            st.switch_page("interfaces/dashboard.py")


# Create a three-column layout for area selection
area_1_col, area_2_col, area_3_col = st.columns(3)

# Create area selection components for each area
create_area_selection("Area 1", area_1_col)
create_area_selection("Area 2", area_2_col)
create_area_selection("Area 3", area_3_col)

# Display footer component
display_footer()