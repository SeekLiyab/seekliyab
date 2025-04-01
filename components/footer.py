import streamlit as st

def display_footer():
    """
    Display a consistent footer across all pages that matches the design shown in the image.
    """
    
    # Footer HTML structure
    st.markdown("""
    <div class="footer-container">
        <div class="logo-container">
            <img src="app/static/images/pup-logo.png" alt="Nursing Logo">
            <img src="app/static/images/pup-ce-logo.png" alt="Mabini Logo">
                <img src="app/static/images/pup-ee-logo.png" alt="CON Logo">
            <img src="app/static/images/seekliyab-logo.png" alt="College Logo">
        </div>
        <div class="footer-title">SEEKLIYAB</div>
        <div class="footer-subtitle">IoT-Based Fire Detection System with Machine Learning</div>
        <div class="footer-college">College of Electrical Engineering</div>
        <div class="footer-links">
            <a href="/about_us">About Us</a>
            <a href="/project">Project</a>
            <a href="/guide">Developer</a>
        </div>
        <div class="footer-copyright">©Polytechnic University of the Philippines | SeekLiyab | All Rights Reserved</div>
    </div>
    """, unsafe_allow_html=True)