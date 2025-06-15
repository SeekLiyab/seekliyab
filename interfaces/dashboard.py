"""
Dashboard interface for the SeekLiyab fire detection system.

This module provides a real-time visualization of sensor data for 
selected areas, including fire risk assessment and historical data.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from components.footer import display_footer
from datetime import timedelta, datetime
from services.utils import get_risk_level_style
from services.database import get_recent_readings_for_area
from services.sms import check_for_sms, check_if_sent, send_sms, send_email


def get_area_from_state_or_params():
    """
    Get the selected area from session state or query parameters.
    
    Returns:
        str or None: The selected area name if available, otherwise None
    """
    selected_area = None
    
    if "selected_area" in st.session_state:
        selected_area = st.session_state.selected_area
    elif "area" in st.query_params:
        selected_area = st.query_params["area"]
        
    return selected_area


# Main dashboard rendering
selected_area = get_area_from_state_or_params()

if selected_area:
    # Add back button
    if st.button("← Back to Visitor Page", type="secondary"):
        st.switch_page("interfaces/visitor.py")
    
    _, area_col, _ = st.columns([1, 8, 1])
    with area_col:
        st.subheader(f"Showing data for {selected_area}")
        
        # Create containers for different parts of the dashboard
        status_container = st.empty()
        data_container = st.empty()
        chart_container = st.empty()
        
        # Define a fragment function to fetch and display real-time data
        @st.fragment(run_every=st.session_state.get("refresh_rate", 1))
        def update_sensor_data():
            """
            Fetch and display real-time sensor data for the selected area.
            
            This function:
            1. Retrieves the latest sensor readings using the database service
            2. Updates the UI with status indicators and data table
            
            The function is run at intervals defined by the refresh_rate setting.
            """
            # Get recent readings using database service
            df = get_recent_readings_for_area(selected_area)
            
            if df is not None and not df.empty:
                # Remove area_name column if it exists
                if 'area_name' in df.columns:
                    df = df.drop('area_name', axis=1)
                
                # Convert timestamp to Philippine Time (PHT)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC').dt.tz_convert('Asia/Manila')
                    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S PHT')
                
                # Get the latest reading for status display
                latest = df.iloc[0]
                risk_level = latest['fire_risk']
                
                # Get style for risk level using the utility function
                style = get_risk_level_style(risk_level)
                
                # Update status container
                with status_container:
                    st.markdown(f"""
                    <div style="background-color:{style['color']}; padding:10px; border-radius:5px; margin-bottom:10px;">
                        <h3 style="color:white; text-align:center;">{style['icon']} Current Status: {risk_level} {style['icon']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Update data container with the dataframe
                with data_container:
                    st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Notification Logs"):
                    # Check if SMS should be sent using the specified flow
                    should_send_sms = check_for_sms(df)
                    if should_send_sms:
                        # Get the latest risk level for checking
                        latest_risk = df.iloc[0]['fire_risk']
                        
                        # Check if SMS was already sent in the last hour for this area/classification
                        already_sent = check_if_sent(selected_area, latest_risk)
                        
                        if not already_sent:
                            # SMS not sent recently, proceed to send
                            sms_result = send_sms(selected_area, latest_risk)
                            
                            if sms_result["sent"]:
                                st.success(f"🚨 {latest_risk} detected! SMS sent to emergency contacts.")
                            elif sms_result["blocked_by_cooldown"]:
                                st.info(f"⏰ {latest_risk} detected! SMS not sent - already notified recently (cooldown active to prevent spam).")
                            else:
                                send_email(selected_area, latest_risk)
                                st.error(f"🚨 {latest_risk} detected but SMS failed to send: {sms_result['reason']}")
                                st.warning("Sent an email to the emergency contacts.")
                        else:
                            # SMS was already sent within the last hour
                            st.info(f"⏰ {latest_risk} detected! SMS not sent - already notified within the last hour for this area and risk level.")
                    else:
                        st.info("No consecutive fire risk pattern detected.")

            else:
                with status_container:
                    st.error(f"No data available for {selected_area}")
        
        # Add a slider to control refresh rate
        refresh_rate = st.sidebar.slider(
            "Refresh rate (seconds)", 
            min_value=1.0, 
            max_value=60.0, 
            value=st.session_state.get("refresh_rate", 5.0),
            step=1.0,
            key="refresh_rate"
        )
        
        # Call the fragment function to start real-time updates
        update_sensor_data()
    
else:
    st.info("No area selected. Please click on an area from the main page.")

display_footer()