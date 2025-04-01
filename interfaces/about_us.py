import streamlit as st
from components.footer import display_footer

_, feed_col, _ = st.columns([1,18,1])

with feed_col:
    # Header with styling
    st.markdown("## 👥 Meet the Team", unsafe_allow_html=True)
    st.markdown("---")

    # Team members data structure for better maintainability
    team_members = [
        {"name": "Billon, Ashlyn Paula I.", "image": "static/images/seekliyab-logo.png", "role": "Electrical Engineering"},
        {"name": "Caringal, Jamilah S.", "image": "static/images/seekliyab-logo.png", "role": "Electrical Engineering"},
        {"name": "Datu, Dexter Daniel E.", "image": "static/images/seekliyab-logo.png", "role": "Electrical Engineering"},
        {"name": "Mazo, Franscine Marie O.", "image": "static/images/seekliyab-logo.png", "role": "Electrical Engineering"},
        {"name": "San Juan, Adrian C.", "image": "static/images/seekliyab-logo.png", "role": "Electrical Engineering"}
    ]

    # Create responsive columns - 3 columns on larger screens, fewer on smaller screens
    cols = st.columns(5)

    # Display team members in a more visually appealing way
    for i, member in enumerate(team_members):
        with cols[i % 5]:
            with st.container(border=True):
                st.image(member["image"], use_container_width=True)
                st.markdown(f"**{member['name']}**")
                st.markdown(f"*{member['role']}*")
                # Add some spacing
                st.write("")

    # About the project with better formatting
    st.markdown("## 🔥 About SeekLiyab")
    st.markdown("""
    The **SeekLiyab** real-time fire detection system was developed by a team of talented BS Electrical Engineering students who combined their expertise in IoT, embedded systems, and machine learning to create an innovative solution for early fire detection and monitoring.

    ### System Components:
    - **Hardware**: Raspberry Pi 4 with multiple sensors
    - MCP9808 temperature sensor
    - MQ135 air quality sensor
    - MQ7 carbon monoxide sensor
    - MQ2 smoke detector
    - **Software**: Custom embedded programming and machine learning algorithms
    - **Interface**: Real-time monitoring dashboard with alert system

    The SeekLiyab project showcases the team's ability to apply their electrical engineering knowledge to develop practical solutions for real-world safety challenges.
    """)

# Call the footer component
display_footer()