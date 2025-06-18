import streamlit as st
from components.footer import display_footer
import os

# Use forward slashes for paths to ensure compatibility with Docker
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
profile_path = os.path.join(root_dir, "static", "images", "profile")

_, main_col, _ = st.columns([1, 16, 1])

with main_col:
    # Header
    st.markdown("# Developer Profile")
    st.markdown("---")
    
    # Profile Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Use a placeholder image or add a professional photo
        st.image(os.path.join(profile_path, "curada.jpg"))
        
    with col2:
        st.markdown("## John Paul (JP) Curada")
        st.markdown("### Python Developer & DataCamp Student Ambassador")
        st.markdown("""
        **Hardworking Innovator with Award-Winning Results & Data Education Advocacy**
        
        A persistent and tenacious **learner** with a proven track record of winning **national and international** 
        data science and solution hackathons. Though still a student, I am driven by genuine curiosity and a relentless 
        work ethic. I grind hard, always exploring new technologies and approaches while specializing in Python development, 
        AI/ML solutions, and IoT systems. **Passionate advocate for data education**, helping others learn and grow in the field.
        """)
        
        # Contact Information
        st.markdown("**Email:** johncurada.02@gmail.com")
        st.markdown("**Location:** Quezon City, Metro Manila")
        st.markdown("**LinkedIn:** [linkedin.com/in/jpcurada](https://linkedin.com/in/jpcurada)")
        st.markdown("**GitHub:** [github.com/JpCurada](https://github.com/JpCurada)")
        st.markdown("**DataCamp:** [datacamp.com/portfolio/jpcurada](https://datacamp.com/portfolio/jpcurada)")

    st.markdown("---")

    # Achievements & Certifications
    st.markdown("## Key Achievements & Certifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Competition Wins (As a Student)**
        - **1st Place** - ASCENT 2023 Data Storytelling Challenge (UP Diliman)
        - **1st Place** - DAP Start Hackathon 2023 (75+ teams)
        - **2nd Place** - SIDHI Data Analysis Hackathon 2023
        - **2nd Place** - PACSiyensya Jr. 2023 (National Science Competition)
        - **Winner** - Eskwelabs Sole Searcher's Hackathon 2023
        - **Finalist** - TechUp Inter University Innovation Challenge 2024
        - **Finalist** - Hack4Health Start Hackathon 2024
        """)
        
    with col2:
        st.markdown("""
        **Certifications & Education Advocacy**
        - **AWS Certified Cloud Practitioner**
        - **DataCamp Certified Python Data Associate**
        - **DataCamp Certified AI Engineer**
        - **DataCamp Student Ambassador** - Promoting data education
        - **Digital Marketing with AI** (ADB & Eskwelabs)
        
        **Current Education**
        - **BS Computer Science** - Polytechnic University of the Philippines (2023-2027)
        - **STEM Strand** - Honorato C. Perez Sr. Memorial Science High School (2021-2023)
        """)

    st.markdown("---")

    # Current Experience
    st.markdown("## Current Experience & Projects")
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        st.markdown("""
        **Freelance Python Developer** *(October 2024 - Present)*
        - Built YOLOv8 real-time fabric defect detection with Streamlit admin interface
        - Developed research library management systems
        - Created IoT fire detection systems with machine learning integration
        - Implemented transfer learning for custom datasets using TensorFlow/Keras
        
        **Data Education & Leadership Roles**
        - **DataCamp Student Ambassador** - Advocating for data education and literacy
        - **Data & ML Lead** - Google Developer Student Clubs PUP (Sept 2024 - Present)
        - **Senior Data Analyst** - AWS Cloud Club PUP (Nov 2023 - Present)
        - **Training Developer** - Created Python for Data Analytics programs
        """)
        
    with exp_col2:
        st.markdown("""
        **Omdena Global Projects** *(August 2023 - June 2024)*
        
        **Junior Machine Learning Engineer (Voluntary)**
        Collaborated with AI/ML engineers worldwide on community impact projects:
        
        - **Personal Injury Claims Valuation (US)**: Conducted EDA and outlier detection using boxplots/whiskers for ML-based legal compensation prediction
        - **Tuberculosis Data Analysis (Nigeria)**: Led data preparation team, automated Excel processing (80% time reduction), developed PyGWalker/Plotly interface
        - **Mango Leaf Disease Detection (Bangladesh)**: Spearheaded EDA and built custom dataset loading functions for Computer Vision agricultural model
        - **Climate Data Analysis (São Paulo, Brazil)**: Engineered weather data retrieval for 645 regions using OpenStreet/Weather APIs
        """)

    st.markdown("---")

    # Data Education Advocacy Section
    st.markdown("## Data Education Advocacy")
    
    advocacy_col1, advocacy_col2 = st.columns(2)
    
    with advocacy_col1:
        st.markdown("""
        **DataCamp Student Ambassador Program**
        - Promoting **data literacy** and **Python education** in academic communities
        - Organizing workshops and training sessions for students
        - Creating educational content and learning materials
        - Bridging the gap between theoretical knowledge and practical application
        """)
        
    with advocacy_col2:
        st.markdown("""
        **Educational Impact & Training**
        - **Python for Data Analytics** training programs for junior analysts
        - Comprehensive learning materials including articles and Jupyter notebooks
        - **Microsoft Youth Ambassador** experience (July 2023 - January 2024)
        - Consistently received commendable feedback for training program clarity
        """)

    st.markdown("---")

    # Featured Projects
    st.markdown("## Featured Student Projects")
    
    projects = [
        {
            "title": "SeekLiyab: IoT Fire Detection System",
            "description": "Comprehensive IoT fire detection with real-time monitoring, ML predictions, and emergency response coordination dashboard built during my studies.",
            "tech": "Python, Streamlit, Supabase, scikit-learn, IoT Sensors",
            "highlight": "Current Featured Project"
        },
        {
            "title": "Diabeathis: Healthcare AI Platform",
            "description": "Innovative diabetes management platform developed for Google's APAC Solution Challenge 2025 international competition.",
            "tech": "TypeScript, React, Node.js, Firebase, Healthcare APIs",
            "highlight": "International Competition"
        },
        {
            "title": "YOLOv8 Fabric Defect Detection",
            "description": "Real-time computer vision system for industrial fabric quality control with Streamlit admin interface - freelance project.",
            "tech": "YOLOv8, Ultralytics, ONNX, Roboflow, Streamlit",
            "highlight": "AI/Computer Vision"
        },
        {
            "title": "Python Data Analytics Training Program",
            "description": "Comprehensive educational program covering Python fundamentals, data structures, and advanced analytics techniques for AWS Cloud Club members.",
            "tech": "Python, Pandas, Jupyter Notebooks, Educational Content",
            "highlight": "Data Education Advocacy"
        },
        {
            "title": "Exploralytics: Python Package",
            "description": "Published Python package for intermediate Plotly visualizations to accelerate data exploration workflows - open source contribution.",
            "tech": "Python, Plotly, setuptools, PyPI",
            "highlight": "Open Source Package"
        },
        {
            "title": "Omdena Global Impact Projects",
            "description": "Multiple international AI/ML projects addressing real-world problems in healthcare, agriculture, and climate analysis across 4 countries.",
            "tech": "Python, TensorFlow, PyGWalker, Plotly, APIs, Computer Vision",
            "highlight": "Global Collaboration"
        }
    ]
    
    for project in projects:
        with st.container(border=True):
            st.markdown(f"### {project['title']} - {project['highlight']}")
            st.markdown(project['description'])
            st.markdown(f"**Technologies:** {project['tech']}")

    st.markdown("---")

    # Services Offered
    st.markdown("## Services I Offer")
    
    services = [
        {
            "title": "Python Development & AI/ML Solutions",
            "description": "Custom Python applications, machine learning models, and computer vision systems with proven competition-winning results"
        },
        {
            "title": "Data Education & Training",
            "description": "Python and data analytics training programs, educational content creation, and mentorship for individuals and organizations"
        },
        {
            "title": "IoT System Development",
            "description": "End-to-end IoT solutions with sensor integration, real-time monitoring, and cloud connectivity for smart systems"
        },
        {
            "title": "Data Analytics & Visualization",
            "description": "Advanced data analysis, interactive dashboards, and business intelligence solutions using modern Python tools and frameworks"
        },
        {
            "title": "Web Application Development",
            "description": "Modern web applications using Streamlit, React, and Python backends with cloud deployment and scalable architecture"
        },
        {
            "title": "Competition-Grade Problem Solving",
            "description": "Hackathon-level solutions and rapid prototyping with a track record of winning national and international competitions"
        }
    ]
    
    for i in range(0, len(services), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            service = services[i]
            with st.container(border=True):
                st.markdown(f"### {service['title']}")
                st.markdown(service['description'])
        
        if i + 1 < len(services):
            with col2:
                service = services[i + 1]
                with st.container(border=True):
                    st.markdown(f"### {service['title']}")
                    st.markdown(service['description'])

    st.markdown("---")

    # Why Choose Me
    st.markdown("## Why Choose This Hardworking Innovator?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Proven Results Despite Being a Student**
        - Multiple **1st place wins** in national competitions while studying
        - **International competition** finalist representing the Philippines
        - Successfully delivered **real-world projects** for clients as a student
        - **Published packages** and global open-source contributions
        
        **Student Advantages**
        - **Fresh perspectives** and innovative approaches to problems
        - **Latest technologies** learned in current academic curriculum
        - **Eager to prove myself** with exceptional dedication to projects
        - **Affordable rates** without compromising on quality
        """)
        
    with col2:
        st.markdown("""
        **Data Education Advocacy & Mentorship**
        - **DataCamp Student Ambassador** promoting data literacy
        - **Educational content creator** with proven training effectiveness
        - **Mentoring experience** helping junior analysts and students
        - **Community impact** through knowledge sharing and skill development
        
        **Global Experience & Work Ethic**
        - **International collaboration** through Omdena projects across 4 countries
        - **Relentless work ethic** - I grind hard to exceed expectations
        - **Continuous learning** - always exploring cutting-edge technologies
        - **Genuine curiosity** drives me to find the best solutions
        """)

    st.markdown("---")

    # Call to Action
    st.markdown("## Ready to Work with a Driven Innovator?")
    
    st.markdown("""
    As a **hardworking PUPian** and **DataCamp Student Ambassador**, I bring **fresh energy**, **innovative thinking**, 
    **educational expertise**, and **proven results** to every project. While I may still be studying, my **competition wins**, 
    **real-world experience**, and **passion for data education** speak for themselves. I'm eager to take on new challenges 
    and deliver exceptional results while helping others learn and grow.
    
    **Let's build something amazing together - with student enthusiasm and professional quality!**
    """)
    
    # Contact buttons
    cta_col1, cta_col2, cta_col3, cta_col4 = st.columns(4)
    
    with cta_col1:
        if st.button("Email Me", type="primary", use_container_width=True):
            st.balloons()
            st.success("Send me an email at: johncurada.02@gmail.com")
            
    with cta_col2:
        if st.button("LinkedIn", use_container_width=True):
            st.success("Connect with me: linkedin.com/in/jpcurada")
            
    with cta_col3:
        if st.button("GitHub", use_container_width=True):
            st.success("Check my code: github.com/JpCurada")
            
    with cta_col4:
        if st.button("Portfolio", use_container_width=True):
            st.success("View my work: datacamp.com/portfolio/jpcurada")

    st.markdown("---")

# Call the footer component
display_footer()
