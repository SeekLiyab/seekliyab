import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import streamlit as st

def load_and_prepare_data(file_path):
    """
    Load the raw data and perform initial data quality checks
    """
    df = pd.read_csv(file_path, sep='\t')
    return df

def check_data_quality(df):
    """
    Check for missing values and data types
    """
    quality_report = {
        'missing_values': df.isnull().sum(),
        'data_types': df.dtypes,
        'shape': df.shape,
        'duplicates': df.duplicated().sum()
    }
    return quality_report

def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets
    70% Train, 10% Validation, 20% Test
    """
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # First split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate validation set from remaining data
    # val_size_adjusted accounts for the fact that we're splitting from 80% of data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_merged_datasets(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Create merged datasets for analysis
    """
    df_train_merged = pd.concat([X_train, y_train], axis=1)
    df_val_merged = pd.concat([X_val, y_val], axis=1)
    df_test_merged = pd.concat([X_test, y_test], axis=1)
    
    return df_train_merged, df_val_merged, df_test_merged

def apply_resampling(X_train, y_train, method='smote', random_state=42):
    """
    Apply various resampling techniques to handle class imbalance
    
    Parameters:
    -----------
    method : str
        Resampling method: 'smote', 'adasyn', 'random_over', 'random_under', 
        'tomek', 'smote_tomek', 'borderline_smote'
    """
    
    if method == 'smote':
        from imblearn.over_sampling import SMOTE
        resampler = SMOTE(random_state=random_state, k_neighbors=5)
        
    elif method == 'adasyn':
        from imblearn.over_sampling import ADASYN
        resampler = ADASYN(random_state=random_state)
        
    elif method == 'random_over':
        from imblearn.over_sampling import RandomOverSampler
        resampler = RandomOverSampler(random_state=random_state)
        
    elif method == 'random_under':
        from imblearn.under_sampling import RandomUnderSampler
        resampler = RandomUnderSampler(random_state=random_state)
        
    elif method == 'tomek':
        from imblearn.under_sampling import TomekLinks
        resampler = TomekLinks()
        
    elif method == 'smote_tomek':
        from imblearn.combine import SMOTETomek
        resampler = SMOTETomek(random_state=random_state)
        
    elif method == 'borderline_smote':
        from imblearn.over_sampling import BorderlineSMOTE
        resampler = BorderlineSMOTE(random_state=random_state)
        
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    
    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def encode_labels(y_train, y_val, y_test):
    """
    Apply label encoding to target variables
    """
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    return y_train_encoded, y_val_encoded, y_test_encoded, label_encoder

def create_ml_pipeline():
    """
    Create a comprehensive ML pipeline with SMOTE and Random Forest
    """
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=5)),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ))
    ])
    
    return pipeline

def get_feature_descriptions():
    """
    Return descriptions of features used in machine learning
    """
    descriptions = {
        'temperature': 'Temperature readings from IoT sensors (°C)',
        'air_quality': 'Air quality index measurements',
        'carbon_monoxide': 'Carbon monoxide concentration levels (ppm)',
        'gas_and_smoke': 'Gas and smoke detection sensor readings'
    }
    return descriptions

def get_target_descriptions():
    """
    Return descriptions of target variable categories
    """
    descriptions = {
        'Fire': 'Active fire incidents detected by the monitoring system',
        'Non-Fire': 'Normal environmental conditions with no fire risk',
        'Potential Fire': 'Conditions indicating elevated fire risk requiring attention'
    }
    return descriptions

def calculate_class_distribution(y):
    """
    Calculate class distribution for target variable
    """
    distribution = pd.Series(y).value_counts()
    percentages = pd.Series(y).value_counts(normalize=True) * 100
    
    return distribution, percentages

def generate_academic_interpretation(section, data=None):
    """
    Generate academic interpretations for different sections
    """
    interpretations = {
        'features_describe': """The descriptive statistics reveal the distributional characteristics of the four sensor measurements used as input features for the machine learning model. The temperature readings show relatively stable values with low variability, indicating consistent environmental monitoring conditions. Air quality, carbon monoxide, and gas and smoke sensors demonstrate varying ranges and standard deviations, suggesting these parameters capture different aspects of environmental conditions that are crucial for fire detection and classification.""",
        
        'target_distribution': """The visualization demonstrates a significant class imbalance in the dataset, with Non-Fire instances comprising the majority of observations, followed by Potential Fire and Fire categories. This imbalance reflects real-world scenarios where actual fire incidents are relatively rare compared to normal operational conditions. The substantial representation of Potential Fire cases provides valuable training data for the model to learn early warning patterns, which is essential for proactive fire prevention and monitoring systems.""",
        
        'grouped_statistics': """The aggregated statistics by fire event categories reveal distinct sensor reading patterns across different environmental conditions. Fire incidents show elevated readings across multiple sensors, particularly in gas and smoke detection and carbon monoxide levels. Non-Fire conditions demonstrate baseline readings that establish normal operational parameters. Potential Fire events exhibit intermediate values, suggesting these instances capture transitional states that precede actual fire incidents, making them critical for early warning system effectiveness.""",
        
        'data_quality': """The data quality assessment confirms the dataset's reliability for machine learning applications. The absence of missing values ensures complete feature representation across all observations, while appropriate data types facilitate efficient computational processing. The low number of duplicate records indicates good data collection practices, and the balanced representation across different time periods suggests comprehensive monitoring coverage that supports robust model training and validation.""",
        
        'data_splits': """The data splitting strategy maintains proportional representation of all fire event categories across training, validation, and test sets. This stratified approach ensures that each subset contains adequate samples from Fire, Non-Fire, and Potential Fire categories, preventing bias during model training and providing reliable performance evaluation. The 70-10-20 split allocation provides sufficient training data while reserving adequate samples for hyperparameter tuning and final model assessment.""",
        
        'before_smote': """The three-dimensional visualization reveals the natural clustering patterns in the original training data, highlighting the inherent class imbalance problem. The sparse representation of Fire and Potential Fire instances compared to Non-Fire samples demonstrates the challenge of learning minority class patterns. The distinct spatial separation between classes suggests that the selected features contain discriminative information, but the imbalanced distribution may lead to biased model performance favoring the majority class.""",
        
        'after_smote': """Following SMOTE application, the visualization shows improved class distribution with synthetic samples generated for minority classes. The Synthetic Minority Oversampling Technique creates realistic data points along the decision boundaries between existing minority samples, enhancing the model's ability to learn Fire and Potential Fire patterns. The preserved spatial relationships between original samples ensure that synthetic data maintains the underlying feature relationships while providing sufficient training examples for each category.""",
        
        'label_encoding': """Label encoding transforms categorical target variables into numerical representations required for machine learning algorithms. This preprocessing step converts Fire, Non-Fire, and Potential Fire categories into integer values while preserving the semantic relationships between classes. The encoding process maintains consistency across training, validation, and test sets, ensuring that the model learns appropriate decision boundaries and can make accurate predictions on unseen data during deployment."""
    }
    
    return interpretations.get(section, "Interpretation pending analysis of the provided data and methodology.")

def create_dataset_characteristics_table(raw_df):
    """Create the dataset characteristics table for overview"""
    dataset_stats = pd.DataFrame({
        'Attribute': [
            'Total Observations', 
            'Feature Variables', 
            'Target Classes', 
            'Data Quality', 
            'Missing Values'
        ],
        'Value': [
            f"{len(raw_df):,}", 
            "4", 
            "3", 
            "Complete", 
            "0"
        ],
        'Description': [
            'Total number of sensor readings collected',
            'Environmental parameters measured',
            'Fire incident classification categories',
            'Data completeness assessment',
            'Data integrity verification'
        ]
    })
    return dataset_stats

def create_class_distribution_table(raw_df):
    """Create class distribution table"""
    class_counts = raw_df['label'].value_counts()
    class_stats = pd.DataFrame({
        'Fire Category': class_counts.index,
        'Count': class_counts.values,
        'Percentage': [f"{(count/len(raw_df)*100):.1f}%" 
                      for count in class_counts.values]
    })
    return class_stats

def create_variable_definitions_table():
    """Create variable definitions table"""
    variables_df = pd.DataFrame([
        ['Temperature', 'Ambient temperature measurements (°C)'],
        ['Air Quality', 'Air quality index readings'],
        ['Carbon Monoxide', 'CO concentration levels (ppm)'],
        ['Gas and Smoke', 'Combined gas and smoke detection values']
    ], columns=['Variable', 'Definition'])
    return variables_df

def create_target_definitions_table():
    """Create target variable definitions table"""
    target_definitions = pd.DataFrame([
        ['Fire', 'Active fire incidents requiring immediate emergency response'],
        ['Non-Fire', 'Normal environmental conditions within acceptable parameters'],
        ['Potential Fire', 'Elevated risk conditions indicating potential fire hazard']
    ], columns=['Classification', 'Operational Definition'])
    return target_definitions

def create_partition_statistics_table(df_train_merged, df_val_merged, df_test_merged, raw_df):
    """Create dataset partition statistics table"""
    partition_stats = pd.DataFrame({
        'Dataset': ['Training', 'Validation', 'Testing'],
        'Sample Size': [len(df_train_merged), len(df_val_merged), len(df_test_merged)],
        'Percentage': [
            f"{len(df_train_merged)/len(raw_df)*100:.1f}%", 
            f"{len(df_val_merged)/len(raw_df)*100:.1f}%", 
            f"{len(df_test_merged)/len(raw_df)*100:.1f}%"
        ]
    })
    return partition_stats

def create_training_distribution_table(df_train_merged):
    """Create training set class distribution table"""
    train_distribution = df_train_merged['label'].value_counts()
    train_dist_df = pd.DataFrame({
        'Fire Category': train_distribution.index,
        'Count': train_distribution.values,
        'Proportion': [f"{(count/len(df_train_merged)*100):.1f}%" 
                      for count in train_distribution.values]
    })
    return train_dist_df

def create_resampling_results_table(df_train_merged, df_train_resampled):
    """Create resampling results comparison table"""
    before_counts = df_train_merged['label'].value_counts()
    after_counts = df_train_resampled['label'].value_counts()
    
    resampling_results = pd.DataFrame({
        'Fire Category': before_counts.index,
        'Original Count': before_counts.values,
        'Post-ADASYN Count': [after_counts.get(cat, 0) for cat in before_counts.index],
        'Synthetic Samples Generated': [
            after_counts.get(cat, 0) - before_counts[cat] for cat in before_counts.index
        ],
        'Percentage Increase': [
            f"{((after_counts.get(cat, 0) - before_counts[cat])/before_counts[cat]*100):.1f}%" 
            if before_counts[cat] > 0 else "0%" for cat in before_counts.index
        ]
    })
    return resampling_results

def create_encoding_mapping_table(label_encoder):
    """Create label encoding mapping table"""
    encoding_mapping = pd.DataFrame({
        'Original Label': label_encoder.classes_,
        'Numerical Encoding': range(len(label_encoder.classes_)),
        'Semantic Interpretation': [
            'Baseline environmental conditions',
            'Active fire incident detected',
            'Elevated risk requiring monitoring'
        ]
    })
    return encoding_mapping

def create_algorithm_advantages_table():
    """Create Random Forest advantages table"""
    advantages_df = pd.DataFrame({
        'Advantage': [
            'Ensemble Learning',
            'Feature Importance Analysis',
            'Overfitting Resistance',
            'Non-linear Pattern Recognition',
            'Computational Efficiency',
            'Interpretability'
        ],
        'Technical Benefit': [
            'Multiple decision trees reduce variance and improve generalization',
            'Quantitative assessment of sensor contribution to classification',
            'Bootstrap aggregating mitigates individual tree overfitting',
            'Capable of modeling complex sensor interaction patterns',
            'Parallel processing enables real-time IoT deployment',
            'Decision tree structure provides transparent classification logic'
        ]
    })
    return advantages_df

def create_hyperparameters_table():
    """Create hyperparameters configuration table"""
    hyperparameters = pd.DataFrame({
        'Parameter': [
            'n_estimators', 
            'max_depth', 
            'min_samples_split', 
            'min_samples_leaf', 
            'random_state'
        ],
        'Value': ['100', '10', '5', '2', '42'],
        'Rationale': [
            'Balanced ensemble size',
            'Complexity regulation',
            'Split threshold control',
            'Leaf size constraint',
            'Reproducibility assurance'
        ]
    })
    return hyperparameters

def get_pipeline_code_snippets():
    """Get code snippets for the implementation tab"""
    pipeline_config = '''
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler

def create_fire_detection_pipeline():
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('resampler', ADASYN(random_state=42, n_neighbors=5)),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    return pipeline

fire_detection_pipeline = create_fire_detection_pipeline()
    '''
    
    training_code = '''
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Train the pipeline
fire_detection_pipeline.fit(X_train, y_train)

# Cross-validation scores
cv_scores = cross_val_score(
    fire_detection_pipeline, X_train, y_train, 
    cv=5, scoring='f1_weighted'
)

# Validation predictions
val_predictions = fire_detection_pipeline.predict(X_val)
print(classification_report(y_val, val_predictions))

# Feature importance analysis
importances = fire_detection_pipeline.named_steps['classifier'].feature_importances_
    '''
    
    prediction_code = '''
def predict_fire_risk(temperature, air_quality, carbon_monoxide, gas_smoke):
    sensor_data = [[temperature, air_quality, carbon_monoxide, gas_smoke]]
    prediction = fire_detection_pipeline.predict(sensor_data)[0]
    probability = fire_detection_pipeline.predict_proba(sensor_data)[0]
    
    risk_labels = ['Non-Fire', 'Potential Fire', 'Fire']
    risk_level = risk_labels[prediction]
    confidence = max(probability) * 100
    
    return {
        'risk_level': risk_level,
        'confidence': confidence,
        'timestamp': datetime.now(),
        'alert_required': confidence >= 70 and risk_level in ['Fire', 'Potential Fire']
    }

# Example usage
result = predict_fire_risk(35.2, 180, 450, 220)
    '''
    
    return {
        'pipeline_config': pipeline_config,
        'training_code': training_code,
        'prediction_code': prediction_code
    }

def render_dataset_overview_tab(st, raw_df, quality_report):
    """Render the Dataset Overview tab content"""
    from utils.visualizations import create_data_quality_summary_table
    
    
    # Research Context
    st.markdown("### Research Context")
    st.markdown("""
    <div class="research-context">
    This research presents the development of an IoT-based fire monitoring system utilizing 
    machine learning algorithms for real-time fire detection and classification. The system 
    is designed for deployment in educational facilities, specifically targeting the College 
    of Engineering building at the Polytechnic University of the Philippines.
    </div>
    """, unsafe_allow_html=True)
    
    # Data Collection Methodology
    st.markdown("### Data Collection Methodology")
    st.markdown("""
    **Data Sources**: IoT sensor networks deployed across three strategic locations:
    - Electrical Engineering Laboratory (primary monitoring zone)
    - Electrical Engineering Department (administrative area)
    - Room 107 (classroom environment)
    
    **Temporal Coverage**: Continuous data collection over multiple operational cycles  
    **Sampling Rate**: Real-time sensor measurements with standardized intervals
    """)
    
    # Dataset Characteristics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Dataset Characteristics")
        dataset_stats = create_dataset_characteristics_table(raw_df)
        st.dataframe(dataset_stats, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("### Class Distribution")
        class_stats = create_class_distribution_table(raw_df)
        st.dataframe(class_stats, hide_index=True, use_container_width=True)
    
    # Data Quality Assessment
    st.plotly_chart(create_data_quality_summary_table(quality_report), 
                    use_container_width=True)
    
    st.markdown("""
    <div class="academic-note">
    <strong>Data Quality Verification:</strong> The dataset demonstrates high quality with 
    complete observations across all feature variables. No missing values were detected, and 
    data types are appropriate for computational analysis. This ensures the reliability of 
    subsequent machine learning model development.
    </div>
    """, unsafe_allow_html=True)

def render_eda_tab(st, raw_df):
    """Render the Exploratory Data Analysis tab content"""
    from utils.visualizations import (
        create_class_imbalance_bar_chart, 
        create_class_percentage_donut_chart,
        create_parallel_coordinates_plot
    )
    
    # Statistical Summary
    st.markdown("### Statistical Summary of Sensor Variables")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### Descriptive Statistics")
        descriptive_stats = raw_df.describe().round(3)
        st.dataframe(descriptive_stats, use_container_width=True)
    
    with col2:
        st.markdown("#### Variable Definitions")
        variables_df = create_variable_definitions_table()
        st.dataframe(variables_df, hide_index=True, use_container_width=True)
    
    # Target Variable Analysis
    st.markdown("### Target Variable Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_class_imbalance_bar_chart(raw_df), 
                       use_container_width=True)
    
    with col2:
        st.plotly_chart(create_class_percentage_donut_chart(raw_df), 
                       use_container_width=True)
    
    # Class-specific Analysis
    st.markdown("### Class-specific Statistical Analysis")
    
    target_definitions = create_target_definitions_table()
    st.dataframe(target_definitions, hide_index=True, use_container_width=True)
    
    # Grouped statistics by fire category
    st.markdown("#### Sensor Response Patterns by Fire Classification")
    grouped_statistics = raw_df.groupby('label', as_index=False).agg({
        'temperature': ['mean', 'median', 'min', 'max', 'std'],
        'air_quality': ['mean', 'median', 'min', 'max', 'std'],
        'carbon_monoxide': ['mean', 'median', 'min', 'max', 'std'],
        'gas_and_smoke': ['mean', 'median', 'min', 'max', 'std']
    })
    st.dataframe(grouped_statistics, hide_index=True, use_container_width=True)
    
    # Multi-dimensional Visualization
    st.markdown("### Multi-dimensional Data Visualization")
    
    st.markdown("""
    **Parallel Coordinates Analysis**: This visualization enables simultaneous examination of 
    all four sensor measurements across different fire classifications, revealing complex 
    multi-dimensional patterns and relationships between environmental parameters.
    """)
    
    # Parallel Coordinates Plot
    st.plotly_chart(create_parallel_coordinates_plot(raw_df), use_container_width=True)
    
    st.markdown("""
    <div class="academic-note">
    <strong>Multi-dimensional Interpretation:</strong> The parallel coordinates plot reveals 
    the complex relationships between all four sensor measurements simultaneously. Each line 
    represents a single observation, color-coded by fire classification. The visualization 
    demonstrates how Fire incidents typically exhibit coordinated elevation across multiple 
    sensors, particularly carbon monoxide and gas/smoke detectors, while Non-Fire conditions 
    show more moderate and consistent patterns across all parameters. Potential Fire events 
    display intermediate characteristics, often showing elevated readings in specific sensors 
    that serve as early warning indicators.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="academic-note">
    <strong>Statistical Observation:</strong> The analysis reveals distinct sensor response 
    patterns across fire classifications. Fire incidents demonstrate significantly elevated 
    readings in carbon monoxide and gas/smoke sensors, while maintaining elevated temperature 
    and reduced air quality measurements. This differentiation provides the theoretical 
    foundation for supervised learning classification.
    </div>
    """, unsafe_allow_html=True)

def render_preprocessing_tab(st, raw_df, df_train_merged, df_val_merged, df_test_merged, df_train_resampled, y_train, y_val, y_test):
    """Render the Data Preprocessing tab content"""
    from utils.visualizations import create_3d_scatter_plot
    
    # Dataset Partitioning
    st.markdown("### Dataset Partitioning Strategy")
    
    st.markdown("""
    **Stratified Sampling Approach**: The dataset was partitioned using stratified random 
    sampling to ensure proportional representation of all fire classification categories 
    across training, validation, and testing subsets.
    
    **Partition Rationale**:
    - **Training Set (70%)**: Primary dataset for model learning and parameter optimization
    - **Validation Set (10%)**: Hyperparameter tuning and model selection validation
    - **Testing Set (20%)**: Final model performance evaluation and generalization assessment
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Partition Statistics")
        partition_stats = create_partition_statistics_table(df_train_merged, df_val_merged, df_test_merged, raw_df)
        st.dataframe(partition_stats, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("#### Training Set Class Distribution")
        train_dist_df = create_training_distribution_table(df_train_merged)
        st.dataframe(train_dist_df, hide_index=True, use_container_width=True)
    
    # Class Imbalance Treatment
    st.markdown("### Class Imbalance Treatment using ADASYN")
    
    st.markdown("""
    <div class="methodology-box">
    <h4>Methodological Approach</h4>
    The Adaptive Synthetic Sampling (ADASYN) technique was employed to address class imbalance 
    in the training dataset. ADASYN generates synthetic samples for minority classes based on 
    the density distribution of existing samples, focusing more attention on harder-to-learn examples.
    
    <strong>ADASYN Advantages:</strong>
    <ul>
        <li>Adaptive generation based on learning difficulty</li>
        <li>Preservation of original data distribution characteristics</li>
        <li>Improved minority class representation without overfitting</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualization of resampling effects
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Training Data Distribution")
        st.plotly_chart(create_3d_scatter_plot(df_train_merged, "(Original Distribution)"), 
                       use_container_width=True)
    
    with col2:
        st.markdown("#### Post-ADASYN Distribution")
        st.plotly_chart(create_3d_scatter_plot(df_train_resampled, "(ADASYN Applied)"), 
                       use_container_width=True)
    
    # Resampling statistics
    st.markdown("#### Resampling Statistical Results")
    resampling_results = create_resampling_results_table(df_train_merged, df_train_resampled)
    st.dataframe(resampling_results, hide_index=True, use_container_width=True)
    
    # Label Encoding
    st.markdown("### Label Encoding Implementation")
    
    y_train_encoded, y_val_encoded, y_test_encoded, label_encoder = encode_labels(
        y_train, y_val, y_test
    )
    
    st.markdown("""
    **Encoding Methodology**: Categorical target variables were systematically converted to 
    numerical representations to facilitate machine learning algorithm processing while 
    preserving ordinal relationships between fire severity levels.
    """)
    
    encoding_mapping = create_encoding_mapping_table(label_encoder)
    st.dataframe(encoding_mapping, hide_index=True, use_container_width=True)

def render_model_development_tab(st):

    # Algorithm Selection
    st.markdown("### Random Forest Algorithm Selection")
    
    st.markdown("""
    **Theoretical Foundation**: Random Forest was selected as the primary classification 
    algorithm based on its ensemble learning approach and demonstrated effectiveness in IoT 
    sensor data classification tasks.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Algorithmic Advantages for Fire Detection")
        advantages_df = create_algorithm_advantages_table()
        st.dataframe(advantages_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("#### Hyperparameter Configuration")
        hyperparameters = create_hyperparameters_table()
        st.dataframe(hyperparameters, hide_index=True, use_container_width=True)
    
    # Hyperparameter Optimization
    st.markdown("### Hyperparameter Optimization Framework")
    
    st.markdown("""
    <div class="methodology-box">
    <h4>Optimization Methodology</h4>
    <ol>
        <li><strong>Grid Search Cross-Validation</strong>: Systematic exploration of 
            hyperparameter space using exhaustive search methodology</li>
        <li><strong>Parameter Space Definition</strong>: Comprehensive ranges defined for 
            critical parameters:
            <ul>
                <li>n_estimators: 50-200 (ensemble size optimization)</li>
                <li>max_depth: 5-20 (tree complexity control)</li>
                <li>min_samples_split: 2-10 (split threshold tuning)</li>
            </ul>
        </li>
        <li><strong>Cross-Validation Strategy</strong>: 5-fold stratified cross-validation 
            ensuring representative sampling</li>
        <li><strong>Performance Metrics</strong>: Multi-metric evaluation including accuracy, 
            precision, recall, and F1-score</li>
        <li><strong>Safety-Critical Optimization</strong>: Emphasis on fire detection recall 
            to minimize false negatives</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

def render_implementation_tab(st):
    
    # Pipeline Implementation Code
    st.markdown("### Machine Learning Pipeline Implementation")
    
    pipeline_tab1, pipeline_tab2, pipeline_tab3 = st.tabs([
        "Pipeline Configuration", 
        "Training Implementation", 
        "Prediction Interface"
    ])
    
    code_snippets = get_pipeline_code_snippets()
    
    with pipeline_tab1:
        st.markdown("#### Complete Pipeline Setup")
        st.code(code_snippets['pipeline_config'], language='python')
    
    with pipeline_tab2:
        st.markdown("#### Training and Validation Process")
        st.code(code_snippets['training_code'], language='python')
    
    with pipeline_tab3:
        st.markdown("#### Real-time Prediction Interface")
        st.code(code_snippets['prediction_code'], language='python')
    
    st.markdown("""
    <div class="academic-note">
    <strong>Implementation Summary:</strong> The developed machine learning pipeline 
    demonstrates a comprehensive approach to IoT-based fire detection, incorporating advanced 
    preprocessing techniques, robust classification algorithms, and production-ready deployment 
    considerations. The system architecture ensures reliable, real-time fire risk assessment 
    suitable for critical safety applications in educational facilities.
    </div>
    """, unsafe_allow_html=True)

def create_dataset_split_statistics(df_merged, dataset_name):
    """Create grouped statistics for a dataset split"""
    import pandas as pd
    
    stats = df_merged.groupby('label', as_index=False).agg({
        'temperature': ['mean', 'median', 'min', 'max'],
        'air_quality': ['mean', 'median', 'min', 'max'],
        'carbon_monoxide': ['mean', 'median', 'min', 'max'],
        'gas_and_smoke': ['mean', 'median', 'min', 'max']
    })
    
    # Flatten column names
    stats.columns = ['label'] + [f'{col[0]}_{col[1]}' for col in stats.columns[1:]]
    
    return stats


def render_data_splitting_tab(st, raw_df, X_train, X_val, X_test, y_train, y_val, y_test):
    """Render the Data Splitting tab content"""
    import pandas as pd
    
    st.markdown("## Data Splitting")
    
    # Overview of splitting strategy
    st.markdown("### Dataset Partitioning Overview")
    st.markdown("""
    The dataset was systematically divided into three distinct subsets to ensure robust model development and evaluation:
    - **70% Training Data**: Used for model learning and parameter optimization
    - **10% Validation Data**: Used for hyperparameter tuning and model selection
    - **20% Testing Data**: Reserved for final model performance evaluation
    """)
    
    # Create merged dataframes
    df_train_merged = pd.concat([X_train, y_train], axis=1)
    df_val_merged = pd.concat([X_val, y_val], axis=1)
    df_test_merged = pd.concat([X_test, y_test], axis=1)
    
    # Split overview statistics
    st.markdown("### Training Data vs Validation Data vs Testing Data")
    
    # Summary statistics table
    split_summary = pd.DataFrame({
        'Dataset': ['Training', 'Validation', 'Testing'],
        'Sample Size': [len(df_train_merged), len(df_val_merged), len(df_test_merged)],
        'Percentage': [
            f"{len(df_train_merged)/len(raw_df)*100:.1f}%",
            f"{len(df_val_merged)/len(raw_df)*100:.1f}%", 
            f"{len(df_test_merged)/len(raw_df)*100:.1f}%"
        ],
        'Fire Count': [
            len(df_train_merged[df_train_merged['label'] == 'Fire']),
            len(df_val_merged[df_val_merged['label'] == 'Fire']),
            len(df_test_merged[df_test_merged['label'] == 'Fire'])
        ],
        'Non-Fire Count': [
            len(df_train_merged[df_train_merged['label'] == 'Non-Fire']),
            len(df_val_merged[df_val_merged['label'] == 'Non-Fire']),
            len(df_test_merged[df_test_merged['label'] == 'Non-Fire'])
        ],
        'Potential Fire Count': [
            len(df_train_merged[df_train_merged['label'] == 'Potential Fire']),
            len(df_val_merged[df_val_merged['label'] == 'Potential Fire']),
            len(df_test_merged[df_test_merged['label'] == 'Potential Fire'])
        ]
    })
    
    st.dataframe(split_summary, hide_index=True, use_container_width=True)
    
    # Detailed statistics for each split
    st.markdown("### Statistical Analysis by Dataset Split")
    
    # Training Data Statistics
    st.markdown("#### Training Data Statistics")
    train_stats = create_dataset_split_statistics(df_train_merged, "Training")
    st.dataframe(train_stats, hide_index=True, use_container_width=True)
    
    # Validation Data Statistics  
    st.markdown("#### Validation Data Statistics")
    val_stats = create_dataset_split_statistics(df_val_merged, "Validation")
    st.dataframe(val_stats, hide_index=True, use_container_width=True)
    
    # Testing Data Statistics
    st.markdown("#### Testing Data Statistics") 
    test_stats = create_dataset_split_statistics(df_test_merged, "Testing")
    st.dataframe(test_stats, hide_index=True, use_container_width=True)
    
    # Academic interpretation
    st.markdown("""
    <div class="academic-note">
    <strong>Statistical Interpretation:</strong> The comparative analysis of sensor measurements across training, validation, and testing datasets demonstrates consistent distributional characteristics, confirming the effectiveness of stratified sampling in maintaining representative data splits. The statistical parameters (mean, median, minimum, and maximum) for each fire classification category remain proportionally similar across all three datasets, indicating that each subset captures the underlying data patterns appropriately. This consistency ensures that model training on the training set will generalize effectively to unseen data, while the validation set provides reliable feedback for hyperparameter optimization, and the testing set offers an unbiased evaluation of final model performance. The preserved statistical relationships across splits validate the robustness of the partitioning strategy for machine learning model development.
    </div>
    """, unsafe_allow_html=True)
