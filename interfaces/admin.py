"""
Admin interface for the SeekLiyab fire detection system.

This module provides an admin dashboard for system management
and monitoring of all fire detection areas and sensors.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime, timedelta
import random
from components.footer import display_footer
import csv
import io

# Apply custom CSS
st.markdown('<style>@import url("styles.css");</style>', unsafe_allow_html=True)

def is_valid_email(email):
    """Validate email format using regex pattern."""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def get_sensor_data(area=None):
    """
    Mock function to get sensor data from database.
    In a real application, this would query the database.
    
    Args:
        area (str, optional): Filter data by area name
        
    Returns:
        pd.DataFrame: Sensor data
    """
    try:
        # Mock data generation - replace with actual database query
        areas = ["Area A", "Area B", "Area C"]
        
        if area and area not in areas:
            # Simulate no data for an area
            return pd.DataFrame()
            
        now = datetime.now()
        data = []
        
        for i in range(100):
            timestamp = now - timedelta(minutes=i*5)
            for a in areas:
                if area and a != area:
                    continue
                data.append({
                    "timestamp": timestamp,
                    "area": a,
                    "temperature": random.uniform(20, 40),
                    "smoke_level": random.uniform(0, 2),
                    "humidity": random.uniform(30, 70)
                })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        # In a real app, you might want to log this error
        st.error(f"Error retrieving sensor data: {str(e)}")
        return pd.DataFrame()

def export_database_to_csv():
    """
    Mock function to export database to CSV.
    In a real application, this would query the database and format the results.
    
    Returns:
        bytes: CSV data as bytes
    """
    try:
        # Mock data - replace with actual database query
        df = get_sensor_data()
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        return csv_buffer.getvalue()
    except Exception as e:
        st.error(f"Error exporting database: {str(e)}")
        return None

def user_management_section():
    """Handle user management section of the admin interface."""
    
    try:
        
        # Display current users in a clean table format
        if hasattr(st.secrets, 'allowed_users') and hasattr(st.secrets.allowed_users, 'emails'):
            users_df = pd.DataFrame({"Email": st.secrets.allowed_users.emails})
            if not users_df.empty:
                st.dataframe(users_df, use_container_width=True)
            else:
                st.info("No users found in the system.")
        else:
            st.warning("User configuration not found.")
    
        col1, col2 = st.columns(2)

        with col1:
            # Add new user
            st.subheader("Add New User")
            new_email = st.text_input("Email address", key="new_email")
            add_user = st.button("Add User", key="add_user_btn")
            
            if add_user:
                if not new_email:
                    st.error("Please enter an email address.")
                elif not is_valid_email(new_email):
                    st.error("Please enter a valid email address.")
                elif new_email in st.secrets.allowed_users.emails:
                    st.warning(f"User {new_email} already exists.")
                else:
                    try:
                        st.secrets.allowed_users.emails.append(new_email)
                        st.success(f"User {new_email} added successfully.")
                    except Exception as e:
                        st.error(f"Failed to add user: {str(e)}")
        
        with col2:
            # Delete user section
            st.subheader("Remove User")
            delete_email = st.text_input("Email to remove", key="delete_email")
            
            if delete_email:
                if delete_email not in st.secrets.allowed_users.emails:
                    st.error(f"User {delete_email} not found.")
                else:
                    # Confirmation before deletion
                    confirm = st.checkbox(f"Confirm deletion of {delete_email}")
                    delete_confirmed = st.button("Delete User", key="delete_confirmed_btn")
                    if delete_confirmed and confirm:
                        try:
                            st.secrets.allowed_users.emails.remove(delete_email)
                            st.success(f"User {delete_email} removed successfully.")
                        except Exception as e:
                            st.error(f"Failed to remove user: {str(e)}")
                    elif delete_confirmed and not confirm:
                        st.warning("Please confirm deletion by checking the box.")
                    
    except Exception as e:
        st.error(f"An error occurred in user management: {str(e)}")

def emergency_contacts_section():
    """Handle emergency contact management section of the admin interface."""
    try:
        # Display current emergency numbers
        if hasattr(st.secrets, 'emergency_contacts') and hasattr(st.secrets.emergency_contacts, 'emergency_numbers'):
            numbers_df = pd.DataFrame({"Contact Number": st.secrets.emergency_contacts.emergency_numbers})
            if not numbers_df.empty:
                st.dataframe(numbers_df, use_container_width=True)
            else:
                st.info("No emergency contacts found in the system.")
        else:
            st.warning("Emergency contacts configuration not found.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Add Number")
            new_number = st.text_input("Contact number", key="new_emergency_number")
            add_number = st.button("Add Number", key="add_emergency_number_btn")
            if add_number:
                if not new_number:
                    st.error("Please enter a contact number.")
                elif new_number in st.secrets.emergency_contacts.emergency_numbers:
                    st.warning(f"Number {new_number} already exists.")
                else:
                    try:
                        st.secrets.emergency_contacts.emergency_numbers.append(new_number)
                        st.success(f"Number {new_number} added successfully.")
                    except Exception as e:
                        st.error(f"Failed to add number: {str(e)}")
        with col2:
            st.subheader("Remove Number")
            delete_number = st.text_input("Number to remove", key="delete_emergency_number")
            if delete_number:
                if delete_number not in st.secrets.emergency_contacts.emergency_numbers:
                    st.error(f"Number {delete_number} not found.")
                else:
                    confirm = st.checkbox(f"Confirm deletion of {delete_number}", key="confirm_delete_emergency_number")
                    delete_confirmed = st.button("Delete Number", key="delete_emergency_number_btn")
                    if delete_confirmed and confirm:
                        try:
                            st.secrets.emergency_contacts.emergency_numbers.remove(delete_number)
                            st.success(f"Number {delete_number} removed successfully.")
                        except Exception as e:
                            st.error(f"Failed to remove number: {str(e)}")
                    elif delete_confirmed and not confirm:
                        st.warning("Please confirm deletion by checking the box.")
    except Exception as e:
        st.error(f"An error occurred in emergency contacts management: {str(e)}")

def sensor_data_visualization():
    """Handle sensor data visualization section."""
    st.subheader("Real-time Sensor Data")
    
    try:
        # Area selection for filtering
        areas = ["All Areas", "Area A", "Area B", "Area C"]
        selected_area = st.selectbox("Select Area", areas)
        
        # Get data based on selection
        filter_area = None if selected_area == "All Areas" else selected_area
        df = get_sensor_data(area=filter_area)
        
        if df.empty:
            st.warning(f"No sensor data available for {selected_area}.")
            return
            
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Temperature", "Smoke Level", "Humidity"])
        
        with tab1:
            st.subheader("Temperature Readings")
            fig = px.line(df, x="timestamp", y="temperature", color="area", 
                         title="Temperature Over Time")
            fig.update_layout(xaxis_title="Time", yaxis_title="Temperature (°C)")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("Smoke Level Readings")
            fig = px.line(df, x="timestamp", y="smoke_level", color="area",
                         title="Smoke Level Over Time")
            fig.update_layout(xaxis_title="Time", yaxis_title="Smoke Level")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            st.subheader("Humidity Readings")
            fig = px.line(df, x="timestamp", y="humidity", color="area",
                         title="Humidity Over Time")
            fig.update_layout(xaxis_title="Time", yaxis_title="Humidity (%)")
            st.plotly_chart(fig, use_container_width=True)
            
        # Real-time updates section using experimental_fragment
        with st.expander("Live Sensor Readings", expanded=True):
            last_readings = df.groupby('area').last().reset_index()
            
            # Create three columns for the most recent readings
            cols = st.columns(3)
            with cols[0]:
                st.metric("Latest Temperature", 
                         f"{last_readings['temperature'].mean():.1f}°C",
                         f"{random.uniform(-2, 2):.1f}°C")
                
            with cols[1]:
                st.metric("Latest Smoke Level", 
                         f"{last_readings['smoke_level'].mean():.2f}",
                         f"{random.uniform(-0.1, 0.1):.2f}")
                
            with cols[2]:
                st.metric("Latest Humidity", 
                         f"{last_readings['humidity'].mean():.1f}%",
                         f"{random.uniform(-5, 5):.1f}%")
                
    except Exception as e:
        st.error(f"Error visualizing sensor data: {str(e)}")

def database_export_section():
    """Handle database export functionality."""
    st.subheader("Database Export")
    
    try:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("Export the entire database as a CSV file.")
            st.write("This will include all sensor readings and configuration data.")
            
        with col2:
            if st.button("Export to CSV"):
                csv_data = export_database_to_csv()
                if csv_data:
                    # Create download button for the CSV
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"seekliyab_export_{timestamp}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Failed to generate export file.")
    except Exception as e:
        st.error(f"Error in database export: {str(e)}")

def render_admin_interface():
    """
    Render the admin interface with appropriate controls and information.
    
    This function sets up the admin dashboard, including headers, controls,
    and content sections for system management.
    """
    # Main header with app styling
    st.title("SeekLiyab Fire Monitoring Platform")
    
    # Main content in tabs
    tab1, tab2, tab3 = st.tabs(["User Management", "Sensor Data", "Database"])
    
    with tab1:
        user_col, emergency_col = st.columns(2)
        with user_col:  
            user_management_section()
        with emergency_col:
            emergency_contacts_section()
        
    with tab2:
        sensor_data_visualization()
        
    with tab3:
        database_export_section()

# Configure sidebar with enhanced visuals and info
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; margin-bottom: 1.5em;'>
        <img src='https://cdn-icons-png.flaticon.com/512/3135/3135715.png' width='80' style='border-radius:50%; box-shadow:0 2px 8px rgba(128,0,0,0.12); margin-bottom:0.5em;'>
        <h2 style='color:#800000; margin-bottom:0.2em;'>Admin Panel</h2>
        <p style='color:#666; font-size:1em; margin-bottom:0.5em;'>Welcome, Administrator!</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**System Quick Stats:**")
    st.metric("Active Sensors", random.randint(10, 20))
    st.metric("Areas Monitored", 3)
    st.metric("Last Update", datetime.now().strftime('%b %d, %Y %I:%M %p'))
    st.markdown("---")
    st.markdown("**How to use the Admin Panel:**")
    st.markdown("""
    - **User Management:** Add or remove admin users who can access the system.
    - **Emergency Contacts:** Manage the phone numbers that will receive fire alerts.
    - **Sensor Data:** View and analyze real-time and historical sensor readings for all areas.
    - **Database:** Export all sensor and configuration data as a CSV file for backup or analysis.
    - **Log out:** Securely exit the admin dashboard.
    """)
    st.markdown("---")
    st.button("Log out", key="logout", on_click=st.logout, use_container_width=True)

# Render the main interface
render_admin_interface()

# Display the footer
display_footer()