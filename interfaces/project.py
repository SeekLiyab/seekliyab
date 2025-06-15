import streamlit as st
from components.footer import display_footer
import pandas as pd
from utils.helpers import (
    load_and_prepare_data,
    check_data_quality,
    split_data,
    create_merged_datasets,
    apply_resampling,
    render_dataset_overview_tab,
    render_eda_tab,
    render_data_splitting_tab,
    render_preprocessing_tab,
    render_model_development_tab,
    render_implementation_tab,
    generate_academic_interpretation
)

import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Primary data path - Excel format as used in training.py
excel_data_path = os.path.join(root_dir, "model-development", "data", "sensor_readings_rows-06_12_25.xlsx")
# Fallback CSV data path
csv_data_path = os.path.join(root_dir, "model-development", "data", "raw_data.csv")

# Main Title and Research Context
st.markdown("""
<div class="main-title">
    <h1>SeekLiyab: IoT-based Fire Monitoring Platform with Random Forest Model</h1>
    <h3>Machine Learning Implementation for Real-time Fire Detection</h3>
    <p>College of Engineering, Polytechnic University of the Philippines</p>
</div>
""", unsafe_allow_html=True)
# Abstract and Research Overview
st.markdown("### Research Objectives & Key Findings")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Accuracy", "98.66%", "±0.46%")
    st.metric("Fire Detection Recall", "96.97%", "Critical for safety")

with col2:
    st.metric("Cross-Validation Score", "98.50%", "5-fold CV")
    st.metric("Out-of-Bag Score", "98.63%", "Unbiased estimate")

with col3:
    st.metric("Test Samples", "523", "Real-world validation")
    st.metric("Training Samples", "2,613", "With SMOTE balancing")

# Data Loading and Preprocessing with Error Handling
@st.cache_data
def load_dataset():
    """Load and preprocess dataset using the same methodology as training.py"""
    
    # Try Excel format first (primary data source)
    raw_df = load_and_prepare_data(excel_data_path)
    
    # Fallback to CSV if Excel fails
    if raw_df is None:
        raw_df = load_and_prepare_data(csv_data_path)
    
    if raw_df is None:
        st.error("Unable to load dataset. Please check data file paths.")
        return None, None, None, None, None, None, None, None, None, None, None
    
    # Data splitting using training.py methodology
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(raw_df)
    
    # Create merged datasets for analysis
    df_train_merged, df_val_merged, df_test_merged = create_merged_datasets(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Apply SMOTE resampling (aligned with training.py which uses SMOTE)
    X_train_resampled, y_train_resampled = apply_resampling(X_train, y_train, method='smote')
    df_train_resampled = pd.concat([
        X_train_resampled, 
        pd.Series(y_train_resampled, name='label')
    ], axis=1)
    
    return (raw_df, X_train, X_val, X_test, y_train, y_val, y_test, 
            df_train_merged, df_val_merged, df_test_merged, df_train_resampled)

# Load dataset with error handling
dataset_result = load_dataset()

if dataset_result[0] is not None:
    (raw_df, X_train, X_val, X_test, y_train, y_val, y_test, 
     df_train_merged, df_val_merged, df_test_merged, df_train_resampled) = dataset_result
    
    # Get data quality report
    quality_report = check_data_quality(raw_df)
    
    # Navigation Tabs with Academic Structure
    st.markdown('<div class="project-tabs">', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Dataset Statistics & Quality", 
        "Statistical Analysis & Distributions",
        "Data Partitioning & Validation",
        "Feature Engineering & Preprocessing", 
        "Model Performance & Metrics",
        "Implementation Results & Validation"
    ])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Render comprehensive tab content
    with tab1:
        render_dataset_overview_tab(st, raw_df, quality_report)
    
    with tab2:
        render_eda_tab(st, raw_df)
    
    with tab3:
        render_data_splitting_tab(st, raw_df, X_train, X_val, X_test, y_train, y_val, y_test)
    
    with tab4:
        render_preprocessing_tab(st, raw_df, df_train_merged, df_val_merged, 
                               df_test_merged, df_train_resampled, y_train, y_val, y_test)
    
    with tab5:
        render_model_development_tab(st)
    
    with tab6:
        render_implementation_tab(st)
        
    
else:
    st.error("Dataset loading failed - check data file paths")
    
    # Data requirements
    requirements_df = pd.DataFrame({
        'Data Source': ['Primary (Excel)', 'Fallback (CSV)'],
        'Format': ['Multiple sheets', 'Tab-separated'],
        'Expected Path': ['model-development/data/sensor_readings_rows-06_12_25.xlsx', 'model-development/data/raw_data.csv'],
        'Required Columns': ['temperature_reading, air_quality_reading, carbon_monoxide_reading, smoke_reading', 'temperature, air_quality, carbon_monoxide, gas_and_smoke']
    })
    
    st.dataframe(requirements_df, hide_index=True, use_container_width=True)

# Footer with Academic Attribution
st.markdown("---")


display_footer()