import streamlit as st
from components.footer import display_footer
import os

# Use forward slashes for paths to ensure compatibility with Docker
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
profile_path = os.path.join(root_dir, "static", "images", "profile")


_, feed_col, _ = st.columns([1,18,1])

with feed_col:
    # Header with styling
    st.markdown("# About SeekLiyab", unsafe_allow_html=True)
    st.markdown("---")

    # Who We Are Section
    st.markdown("## Who We Are")
    st.markdown("""
    **SeekLiyab** is an innovative fire detection and alarm system that harnesses Internet of Things (IoT) technology for real-time monitoring and instant alerts, enhancing fire safety in homes, businesses, and public spaces. Driven by a team of passionate innovators, the project was born from a critical question: *How can fire detection and response be improved to prevent disasters?*

    With a commitment to fast, reliable, and accessible fire prevention, SeekLiyab provides a smarter, more proactive approach to protecting lives and properties. By integrating advanced technology with a mission to make a real impact, the team behind SeekLiyab continues to push the boundaries of fire safety innovation—turning their vision into reality, one breakthrough at a time.
    """)

    # Background of the Study
    st.markdown("## Background of the Study")
    st.markdown("""
    The SeekLiyab project was developed to address the growing need for improved fire safety systems due to the increasing number of fire incidents caused by delayed detection and response. Traditional fire alarms are often limited by outdated technology, lack of remote access, and slow alerts. 

    In response, SeekLiyab integrates modern IoT technology to enhance fire detection, real-time monitoring, and emergency response. The system uses IoT sensors and automated alerts to offer a proactive approach to fire prevention and control. This innovative project aims to enhance public safety, reduce property damage, and minimize the risks of fire-related disasters by applying emerging technologies in fire safety management.

    The study is driven by a commitment to enhance public safety, minimize property damage, and reduce the risks associated with fire-related disasters. Through extensive research, evaluation of existing systems, and application of emerging technologies, the researchers have developed a solution that is both innovative and practical.
    """)

    # Service and Product
    st.markdown("## Our Solution")
    st.markdown("""
    The SeekLiyab project is an innovative fire detection and alarm system that uses IoT technology to provide real-time monitoring, early fire detection, and immediate alerts. It integrates smart sensors to detect smoke, heat, gas, and temperature, transmitting data to a centralized platform. 

    Through IoT connectivity, users receive instant alerts via mobile apps, enabling prompt action even when away from the premises. SeekLiyab aims to modernize fire safety by offering a more responsive, automated, and accessible solution for homes, businesses, and industrial settings.

    ### System Components:
    - **Hardware**: Raspberry Pi 4 with multiple sensors
      - MCP9808 temperature sensor
      - MQ135 air quality sensor
      - MQ7 carbon monoxide sensor
      - MQ2 smoke detector
    - **Software**: Custom embedded programming and machine learning algorithms
    - **Interface**: Real-time monitoring dashboard with alert system
    """)

    # Meet the Team
    st.markdown("## Meet the Team")
    st.markdown("---")

    # Team members data structure with detailed bios
    team_members = [
        {
            "name": "San Juan, Adrian C.",
            "image": os.path.join(profile_path, "san_juan.jpg"),
            "role": "Electrical Engineering Student",
            "bio": "Adrian San Juan, a 23-year-old graduating Electrical Engineering student, is passionate about science, technology, and leadership. As a former Executive Vice President of a student organization, he honed his skills beyond academics. He aspires to contribute to power generation and sustainable energy, striving to make a meaningful impact in his field."
        },
        {
            "name": "Datu, Dexter Daniel E.",
            "image": os.path.join(profile_path, "datu.jpg"),
            "role": "Electrical Engineering Student",
            "bio": "Mr. Datu is an Electrical Engineering student and a proud Iskolar ng Bayan, driven by a passion for innovation and problem-solving. With a strong interest in power systems, instrumentation, and control, he is determined to contribute to energy efficiency while addressing safety concerns on campus. Through his work, he aims to give back to Pamantasan by applying engineering solutions that create a lasting impact. Outside of academics, he enjoys hands-on projects, exploring new technologies, and engaging in meaningful discussions about the future of engineering."
        },
        {
            "name": "Mazo, Franscine Marie O.",
            "image": os.path.join(profile_path, "mazo.jpg"),
            "role": "Electrical Engineering Student",
            "bio": "Franscine Marie O. Mazo, an Electrical Engineering student, is dedicated to research, documentation, and technical support. With a keen attention to detail, she ensures clarity and accuracy in presenting research findings. While primarily focused on organizing and structuring reports, she also assists in fundamental technical tasks. She aims to bridge technical concepts with effective documentation, enhancing research with clear and meaningful insights in her field."
        },
        {
            "name": "Caringal, Jamilah S.",
            "image": os.path.join(profile_path, "caringal.png"),
            "role": "Electrical Engineering Student",
            "bio": "Jamilah Caringal is an Electrical Engineering student and a committed researcher with a passion for innovation and sustainability. She has a strong interest in renewable energy, automation, and mechatronics, focusing on the development of efficient engineering solutions that support a more sustainable future. Her work is dedicated to tackling energy efficiency challenges and enhancing safety standards through practical and research-driven approaches. She actively explores emerging technologies and participates in discussions on the advancement of electrical engineering."
        },
        {
            "name": "Billon, Ashlyn Paula I.",
            "image": os.path.join(profile_path, "billon.jpg"),
            "role": "Electrical Engineering Student",
            "bio": "Ashlyn Paula Billon is a 21-year-old fourth-year electrical engineering student at PUP Manila. She has a strong foundation in power systems and grid operations, which she further developed during her internship at the National Grid Corporation of the Philippines (NGCP). Passionate about innovation and infrastructure, she applies her technical skills to real-world engineering challenges while continuously expanding her expertise in the field."
        }
    ]

    # Display team members in a more detailed format
    for i, member in enumerate(team_members):
        with st.container(border=True):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image(member["image"], use_container_width=True)
            
            with col2:
                st.markdown(f"### {member['name']}")
                st.markdown(f"**{member['role']}**")
                st.markdown(member['bio'])
                
        st.write("")  # Add spacing between team members

    # Project Impact
    st.markdown("## Project Impact")
    st.markdown("""
    The SeekLiyab project showcases the team's ability to apply their electrical engineering knowledge to develop practical solutions for real-world safety challenges. This innovative system represents a significant step forward in fire safety technology, combining academic excellence with practical application to create solutions that can save lives and protect property.

    Through SeekLiyab, the team demonstrates how emerging technologies can be leveraged to address critical safety concerns while making fire prevention more accessible and efficient for communities.
    """)

    st.markdown("---")


# Call the footer component
display_footer()