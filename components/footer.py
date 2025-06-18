import streamlit as st

def display_footer():
    """
    Display a consistent footer across all pages that matches the design shown in the image.
    """
    
    # Footer HTML structure
    st.markdown("""
    <div class="footer-container">
        <div class="logo-container">
            <img src="app/static/images/pup-logo.png" alt="PUP Logo">
            <img src="app/static/images/pup-ce-logo.png" alt="College of Engineering Logo">
            <img src="app/static/images/pup-ee-logo.png" alt="College of Electrical Engineering Logo">
            <img src="app/static/images/seekliyab-logo.png" alt="Seekliyab Logo">
        </div>
        <div class="footer-title">
            <a href="/" class="footer-title-link">SEEKLIYAB</a>
        </div>
        <div class="footer-subtitle">IoT-Based Fire Detection System with Machine Learning</div>
        <div class="footer-college">College of Electrical Engineering</div>
        <div class="footer-links">
            <a href="/about_us">About Us</a>
            <a href="/project">Project</a>
            <a href="/developer">Developer</a>
        </div>
        <div class="footer-copyright">©Polytechnic University of the Philippines | SeekLiyab | All Rights Reserved</div>
    </div>
    """, unsafe_allow_html=True)