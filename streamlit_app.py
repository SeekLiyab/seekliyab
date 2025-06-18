import streamlit as st
from services.auth_service import login_page
import os

# Use forward slashes for paths to ensure compatibility with Docker
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "static", "images", "seekliyab-logo.png")
css_path = os.path.join(parent_dir, "static", "styles.css")

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
        page_icon=logo_path
    )

    # Load custom CSS
    with open(css_path) as css:
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
    developer = st.Page("interfaces/developer.py", title="Developer")

    # Set up navigation based on authentication status and authorization
    is_authenticated = getattr(st.experimental_user, "is_logged_in", False)
    is_authorized = is_authenticated and st.experimental_user.email in st.secrets.allowed_users.emails
    
    if not is_authorized:
        # Only show login page if not authenticated or not authorized
        return st.navigation([login, visitor, about_us, dashboard, project, developer], position="hidden")  
    else:
        # Show all pages if authenticated and authorized
        return st.navigation([admin, about_us])


# Main application entry point
configure_app()
navigation = setup_navigation()
navigation.run()