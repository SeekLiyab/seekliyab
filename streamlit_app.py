"""
Main application entry point for the SeekLiyab fire detection system.

This module configures the Streamlit application, sets up navigation,
and manages page routing based on authentication status.
"""

import streamlit as st
import os
from services.auth_service import login_page


def configure_app():
    """
    Configure the Streamlit application with appropriate settings.
    
    Sets page title, layout, sidebar state, and loads custom CSS.
    """
    # Page configuration
    st.set_page_config(
        page_title="SeekLiyab", 
        layout="wide", 
        initial_sidebar_state="collapsed", 
        page_icon="app/static/images/seekliyab-logo.png"
    )

    # Load custom CSS
    with open("static/styles.css") as css:
        st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)


def setup_navigation():
    """
    Set up application navigation based on authentication status.
    
    Returns:
        streamlit.navigation: The configured navigation object
    """
    # Define available pages
    login = st.Page(login_page, title="Login")
    admin = st.Page("interfaces/admin.py", title="Admin", default=True)
    visitor = st.Page("interfaces/visitor.py", title="Visitor")
    about_us = st.Page("interfaces/about_us.py", title="About Us")
    dashboard = st.Page("interfaces/dashboard.py", title="Dashboard")
    project = st.Page("interfaces/project.py", title="Project")

    # Set up navigation based on authentication status and authorization
    is_authenticated = getattr(st.experimental_user, "is_logged_in", False)
    is_authorized = is_authenticated and st.experimental_user.email in st.secrets.allowed_users.emails
    
    if not is_authorized:
        # Only show login page if not authenticated or not authorized
        return st.navigation([login, visitor, about_us, dashboard, project], position="hidden")  
    else:
        # Show all pages if authenticated and authorized
        return st.navigation([admin, about_us])


# Main application entry point
configure_app()
navigation = setup_navigation()
navigation.run()