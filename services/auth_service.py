"""
Authentication service for the SeekLiyab application.

This module handles user authentication, including login, logout, and access control.
"""

import streamlit as st
import time
from components.footer import display_footer

def login_page():
    """
    Render the application login page with options for admin and visitor access.
    
    This function manages:
    - Authentication state checking
    - Authorized email validation
    - Admin login form
    - Visitor access button
    - User feedback for invalid credentials
    
    Returns:
        None: The page is rendered directly to the Streamlit app.
    """
    # Check if authentication is required first
    if "auth_required" in st.session_state and st.session_state.auth_required:
        # Clear the flag to prevent infinite loops
        st.session_state.pop("auth_required")
        
        # Initiate Google login directly
        st.login()
        st.stop()

    # Safely check if user is logged in
    is_logged_in = getattr(st.experimental_user, "is_logged_in", False)
    user_email = getattr(st.experimental_user, "email", "")
    
    if is_logged_in and user_email not in st.secrets.allowed_users.emails:
        st.error("Access denied. Your email is not authorized to access the admin area.")
        time.sleep(4)
        st.logout()
        st.stop()
    
    _, login_col, _ = st.columns([1,1,1])

    with login_col:
        login_container = st.container(border=False, key='login_container')
        with login_container:
            st.image("static\images\seekliyab-banner-f.png", use_container_width=True)
            
            login_form = st.form(key="login_form", border=False)
            with login_form:
                username = st.text_input("Username", placeholder="Administator ID")
                password = st.text_input("Password", type="password", placeholder="Access key")
                submit_button = st.form_submit_button("Login as Admin", use_container_width=True, type="primary")
            
            # Form submission handling outside the form
            if submit_button:
                if username == "admin" and password == "seekliyab2025#":
                    show_google_login()
                else:
                    st.toast("Invalid credentials. Please try again.")
            
            st.divider()
            with st.container():
                if st.button("Login as Visitor", use_container_width=True, type="primary"):
                    st.switch_page("interfaces/visitor.py")
                    
            st.caption("No admin account yet? Contact the SeekLiyab team to get started.")
    display_footer()


@st.dialog("Access Confirmation")
def show_google_login():
    """
    Display a dialog prompting the user to confirm Google authentication.
    
    This dialog appears after successful username/password validation to 
    require additional Google account authentication for admin access.
    
    Returns:
        None: The dialog is rendered directly in the Streamlit app.
    """
    st.write("To protect the integrity of the fire data of the institution, you need to log in to the college account to access admin account.")
    if st.button("Proceed with Google Login", type="primary"):
        # Set flag and close dialog
        st.session_state.auth_required = True
        st.rerun()