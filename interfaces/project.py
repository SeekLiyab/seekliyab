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
    render_implementation_tab
)

# Main Title
st.markdown("""
<div class="main-title">
    <h1>SeekLiyab: IoT-based Fire Monitoring Platform with Random Forest Model</h1>
    <h3>Machine Learning Implementation for Real-time Fire Detection</h3>
    <p>College of Engineering, Polytechnic University of the Philippines</p>
</div>
""", unsafe_allow_html=True)

# Data Loading and Preprocessing
@st.cache_data
def load_dataset():
    """Load and preprocess dataset for analysis"""
    raw_df = load_and_prepare_data(r"model-development\data\raw_data.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(raw_df)
    df_train_merged, df_val_merged, df_test_merged = create_merged_datasets(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    X_train_resampled, y_train_resampled = apply_resampling(X_train, y_train, method='adasyn')
    df_train_resampled = pd.concat([
        X_train_resampled, 
        pd.Series(y_train_resampled, name='label')
    ], axis=1)
    
    return (raw_df, X_train, X_val, X_test, y_train, y_val, y_test, 
            df_train_merged, df_val_merged, df_test_merged, df_train_resampled)

# Load dataset
(raw_df, X_train, X_val, X_test, y_train, y_val, y_test, 
 df_train_merged, df_val_merged, df_test_merged, df_train_resampled) = load_dataset()

# Get data quality report
quality_report = check_data_quality(raw_df)

# Navigation Tabs with custom styling
st.markdown('<div class="project-tabs">', unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dataset Overview", 
    "Exploratory Data Analysis",
    "Data Splitting",
    "Data Preprocessing", 
    "Model Development",
    "Implementation & Results"
])
st.markdown('</div>', unsafe_allow_html=True)

# Render tab content using helper functions
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

# Footer
st.markdown("---")
display_footer()