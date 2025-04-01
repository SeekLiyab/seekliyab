import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from components.footer import display_footer


area_1_col, area_2_col, area_3_col = st.columns(3)

# Container for Area 1
with area_1_col.container(border=True):
    area_1_click = streamlit_image_coordinates("static/images/seekliyab-banner-f.png", key="area1", use_column_width=True)
    if area_1_click:
        # Store the area in session state before switching pages
        st.session_state.selected_area = "Area 1"
        # Update query params
        st.query_params["area"] = "Area 1"
        # Switch to dashboard page
        st.switch_page("interfaces/dashboard.py")

# Container for Area 2
with area_2_col.container(border=True):
    area_2_click = streamlit_image_coordinates("static/images/seekliyab-banner-f.png", key="area2", use_column_width=True)
    if area_2_click:
        st.session_state.selected_area = "Area 2"
        st.query_params["area"] = "Area 2"
        st.switch_page("interfaces/dashboard.py")

# Container for Area 3
with area_3_col.container(border=True):
    area_3_click = streamlit_image_coordinates("static/images/seekliyab-banner-f.png", key="area3", use_column_width=True)
    if area_3_click:
        st.session_state.selected_area = "Area 3"
        st.query_params["area"] = "Area 3"
        st.switch_page("interfaces/dashboard.py")

display_footer()