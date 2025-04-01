import streamlit as st
import pandas as pd
from components.footer import display_footer

st.title("Area Dashboard")

# Create sample data for tables
table_data = {
    "Area 1": pd.DataFrame({
        "Column 1": [1, 2, 3],
        "Column 2": ["A", "B", "C"]
    }),
    "Area 2": pd.DataFrame({
        "Product": ["Widget", "Gadget", "Tool"],
        "Price": [10.99, 15.50, 7.25]
    }),
    "Area 3": pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Value": [100, 150, 200]
    })
}

# Try to get area from session state first, then from query params
selected_area = None

if "selected_area" in st.session_state:
    selected_area = st.session_state.selected_area
elif "area" in st.query_params:
    selected_area = st.query_params["area"]

if selected_area:
    st.subheader(f"Showing data for {selected_area}")
    
    # Display the data for the selected area
    if selected_area in table_data:
        st.dataframe(table_data[selected_area])
    else:
        st.error(f"No data available for {selected_area}")
else:
    st.info("No area selected. Please click on an area from the main page.")

display_footer()