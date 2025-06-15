# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.figure_factory as ff
# from plotly.subplots import make_subplots
# import graphviz
# from sklearn.model_selection import (train_test_split, StratifiedKFold,
#                                    RandomizedSearchCV, validation_curve, learning_curve)
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import (classification_report, confusion_matrix,
#                            precision_recall_curve, f1_score, recall_score,
#                            precision_score, accuracy_score, roc_auc_score, roc_curve)
# from sklearn.tree import export_graphviz
# from imblearn.over_sampling import SMOTE
# from sklearn.utils.class_weight import compute_class_weight
# from openpyxl import load_workbook
# import warnings
# warnings.filterwarnings('ignore')


# import pickle


# class SeekLiyabFireDetector:
#     """
#     Production-ready fire detection system for SeekLiyab IoT platform
#     Implements proper data splitting to prevent data leakage and comprehensive model evaluation
#     """

#     def __init__(self, apply_feature_engineering=False):
#         self.model = None
#         self.label_encoder = LabelEncoder()
#         self.feature_stats = {}
#         self.feature_names = None
#         self.class_weights = None
#         self.best_params = None
#         self.evaluation_results = {}
#         self.apply_feature_engineering = apply_feature_engineering

#     def load_data(self, RAW_DATA_FILE_PATH, SENSORS_COLUMN_NAMES):
#         """Load the SeekLiyab sensor data from Excel workbook with multiple sheets"""
#         try:
#             raw_excel_wb = load_workbook(filename=RAW_DATA_FILE_PATH)
#             raw_df = pd.DataFrame()

#             for excel_file_name in raw_excel_wb.sheetnames:
#                 print(f"\nProcessing sheet: {excel_file_name}")

#                 label = excel_file_name.split('_')[0]
#                 print(f"Extracted label: {label}")

#                 if label in ['maybe fire', 'fire', 'non fire']:
#                     print(f"Reading data for label: {label}")
#                     temp_df = pd.read_excel(RAW_DATA_FILE_PATH, sheet_name=excel_file_name)
#                     temp_df['label'] = label

#                     print(f"DataFrame shape before concat: raw_df={raw_df.shape}, temp_df={temp_df.shape}")
#                     raw_df = pd.concat([raw_df, temp_df[SENSORS_COLUMN_NAMES + ['label']]])
#                     print(f"DataFrame shape after concat: raw_df={raw_df.shape}")
#                 else:
#                     print(f"Skipping sheet: {excel_file_name} (label '{label}' not in allowed list)")

#             print(f"Dataset loaded successfully: {raw_df.shape[0]} samples, {raw_df.shape[1]} features")
#             print(f"Features: {list(raw_df.columns)}")
#             print(f"\nClass distribution:")
#             print(raw_df['label'].value_counts())
#             print(f"\nClass percentages:")
#             print(raw_df['label'].value_counts(normalize=True) * 100)
#             return raw_df
#         except Exception as e:
#             print(f"Error loading data: {e}")
#             return None

#     def split_data(self, df, test_size=0.2, val_size=0.2, random_state=42):
#         """
#         Split data into train/validation/test sets FIRST to prevent data leakage
#         """
#         X = df.drop('label', axis=1)
#         y = df['label']

#         # First split: separate test set
#         X_temp, X_test, y_temp, y_test = train_test_split(
#             X, y, test_size=test_size, stratify=y, random_state=random_state
#         )

#         # Second split: separate train and validation from remaining data
#         val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
#         X_train, X_val, y_train, y_val = train_test_split(
#             X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=random_state
#         )

#         print(f"\nData split completed:")
#         print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
#         print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
#         print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

#         return X_train, X_val, X_test, y_train, y_val, y_test

#     def fit_feature_engineering(self, X_train):
#         """Learn feature engineering parameters from training data only"""
#         if not self.apply_feature_engineering:
#             print("Feature engineering is disabled")
#             return

#         # Map column names to standard names
#         temp_col = 'temperature_reading' if 'temperature_reading' in X_train.columns else 'temperature'
#         air_col = 'air_quality_reading' if 'air_quality_reading' in X_train.columns else 'air_quality'
#         co_col = 'carbon_monoxide_reading' if 'carbon_monoxide_reading' in X_train.columns else 'carbon_monoxide'
#         smoke_col = 'smoke_reading' if 'smoke_reading' in X_train.columns else 'gas_and_smoke'

#         self.feature_stats = {
#             'temp_mean': X_train[temp_col].mean(),
#             'temp_std': X_train[temp_col].std(),
#             'temp_q25': X_train[temp_col].quantile(0.25),
#             'temp_q75': X_train[temp_col].quantile(0.75),
#             'air_mean': X_train[air_col].mean(),
#             'air_std': X_train[air_col].std(),
#             'co_mean': X_train[co_col].mean(),
#             'co_std': X_train[co_col].std(),
#             'smoke_mean': X_train[smoke_col].mean(),
#             'smoke_std': X_train[smoke_col].std(),
#         }
#         print("Feature engineering parameters learned from training data")

#     def engineer_features(self, X, is_training=False):
#         """Create engineered features using only training statistics"""
#         if not self.apply_feature_engineering:
#             if is_training:
#                 print("Feature engineering disabled - using original features only")
#             return X.copy()

#         if not self.feature_stats:
#             raise ValueError("Must call fit_feature_engineering() first on training data")

#         X_engineered = X.copy()
#         stats = self.feature_stats

#         # Map column names to standard names for feature engineering
#         temp_col = 'temperature_reading' if 'temperature_reading' in X.columns else 'temperature'
#         air_col = 'air_quality_reading' if 'air_quality_reading' in X.columns else 'air_quality'
#         co_col = 'carbon_monoxide_reading' if 'carbon_monoxide_reading' in X.columns else 'carbon_monoxide'
#         smoke_col = 'smoke_reading' if 'smoke_reading' in X.columns else 'gas_and_smoke'

#         # Temperature-based features (critical for fire detection)
#         X_engineered['temp_zscore'] = (X[temp_col] - stats['temp_mean']) / stats['temp_std']
#         X_engineered['temp_critical'] = (X[temp_col] > 50).astype(int)
#         X_engineered['temp_extreme'] = (X[temp_col] > 60).astype(int)

#         # Gas concentration features
#         X_engineered['co_zscore'] = (X[co_col] - stats['co_mean']) / stats['co_std']
#         X_engineered['smoke_zscore'] = (X[smoke_col] - stats['smoke_mean']) / stats['smoke_std']
#         X_engineered['air_zscore'] = (X[air_col] - stats['air_mean']) / stats['air_std']

#         # Interaction features (physics-based)
#         X_engineered['temp_co_interaction'] = X[temp_col] * X[co_col] / 1000
#         X_engineered['temp_smoke_interaction'] = X[temp_col] * X[smoke_col] / 1000
#         X_engineered['co_smoke_ratio'] = X[co_col] / (X[smoke_col] + 1)

#         # Fire risk composite score
#         temp_risk = np.clip((X[temp_col] - 20) / 50, 0, 1)
#         co_risk = np.clip((X[co_col] - 300) / 500, 0, 1)
#         smoke_risk = np.clip((X[smoke_col] - 200) / 400, 0, 1)
#         air_risk = np.clip((X[air_col] - 300) / 500, 0, 1)

#         X_engineered['fire_risk_composite'] = (
#             0.4 * temp_risk +
#             0.25 * co_risk +
#             0.25 * smoke_risk +
#             0.1 * air_risk
#         )

#         # Anomaly indicators
#         X_engineered['temp_anomaly'] = (abs(X_engineered['temp_zscore']) > 2).astype(int)
#         X_engineered['co_anomaly'] = (abs(X_engineered['co_zscore']) > 2).astype(int)
#         X_engineered['smoke_anomaly'] = (abs(X_engineered['smoke_zscore']) > 2).astype(int)
#         X_engineered['total_anomalies'] = (X_engineered['temp_anomaly'] +
#                                          X_engineered['co_anomaly'] +
#                                          X_engineered['smoke_anomaly'])

#         if is_training:
#             print(f"Feature engineering completed. Shape: {X_engineered.shape}")
#             print(f"Added {X_engineered.shape[1] - X.shape[1]} engineered features")

#         return X_engineered

#     def handle_class_imbalance(self, X_train, y_train, strategy='smote_balanced'):
#         """Handle class imbalance using SMOTE with realistic sampling"""
#         print(f"Original training class distribution:")
#         print(y_train.value_counts())
#         print(f"Original percentages:")
#         print(y_train.value_counts(normalize=True) * 100)

#         if strategy == 'weights_only':
#             return X_train, y_train

#         # Apply SMOTE with balanced sampling
#         y_train_temp = self.label_encoder.fit_transform(y_train)

#         if strategy == 'smote_balanced':
#             # Balanced sampling for all classes
#             smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.value_counts().min()-1))
#             X_resampled, y_resampled = smote.fit_resample(X_train, y_train_temp)
#         else:
#             # Custom sampling strategy
#             unique, counts = np.unique(y_train_temp, return_counts=True)
#             majority_count = max(counts)
#             sampling_strategy = {
#                 cls: min(majority_count, count * 3)
#                 for cls, count in zip(unique, counts)
#             }
#             smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42,
#                          k_neighbors=min(5, y_train.value_counts().min()-1))
#             X_resampled, y_resampled = smote.fit_resample(X_train, y_train_temp)

#         # Convert back to original labels
#         y_resampled_labels = self.label_encoder.inverse_transform(y_resampled)
#         y_resampled_series = pd.Series(y_resampled_labels)

#         print(f"\nAfter SMOTE resampling:")
#         print(y_resampled_series.value_counts())
#         print(f"Resampled percentages:")
#         print(y_resampled_series.value_counts(normalize=True) * 100)

#         return X_resampled, y_resampled_series

#     def encode_labels(self, y_train, y_val=None, y_test=None):
#         """Encode labels using training data mapping"""
#         y_train_encoded = self.label_encoder.fit_transform(y_train)

#         print(f"\nLabel encoding mapping:")
#         for i, label in enumerate(self.label_encoder.classes_):
#             print(f"  {label} -> {i}")

#         results = [y_train_encoded]

#         if y_val is not None:
#             y_val_encoded = self.label_encoder.transform(y_val)
#             results.append(y_val_encoded)

#         if y_test is not None:
#             y_test_encoded = self.label_encoder.transform(y_test)
#             results.append(y_test_encoded)

#         return results if len(results) > 1 else results[0]

#     def compute_class_weights(self, y_train_encoded):
#         """Compute balanced class weights for safety-focused fire detection"""
#         classes = np.unique(y_train_encoded)
#         base_weights = compute_class_weight('balanced', classes=classes, y=y_train_encoded)

#         # Safety calibration: prioritize fire detection
#         safety_weights = {}
#         for i, class_id in enumerate(classes):
#             if self.label_encoder.classes_[class_id] == 'fire':
#                 safety_weights[class_id] = base_weights[i] * 1.5  # Increase fire detection weight
#             elif self.label_encoder.classes_[class_id] == 'maybe fire':
#                 safety_weights[class_id] = base_weights[i] * 1.2  # Moderate increase
#             else:
#                 safety_weights[class_id] = base_weights[i]

#         self.class_weights = safety_weights
#         print(f"Safety-calibrated class weights: {self.class_weights}")
#         return self.class_weights

#     def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, cv_folds=5):
#         """Comprehensive hyperparameter tuning using RandomizedSearchCV"""
#         print("Starting comprehensive hyperparameter optimization...")

#         # Extended parameter space for thorough search
#         param_distributions = {
#             'n_estimators': [100, 200, 300, 500, 800],
#             'max_depth': [10, 15, 20, 25, 30, None],
#             'min_samples_split': [2, 5, 10, 15, 20],
#             'min_samples_leaf': [1, 2, 4, 8, 12],
#             'max_features': ['sqrt', 'log2', 0.6, 0.7, 0.8],
#             'bootstrap': [True, False],
#             'max_samples': [0.7, 0.8, 0.9, None],
#             'criterion': ['gini', 'entropy']
#         }

#         # Base model with class weights
#         rf_base = RandomForestClassifier(
#             class_weight=self.class_weights,
#             random_state=42,
#             n_jobs=-1,
#             oob_score=True
#         )

#         # Custom scoring function prioritizing fire safety
#         def fire_safety_scorer(estimator, X, y):
#             y_pred = estimator.predict(X)

#             # Multi-objective scoring: balance recall and precision for fire classes
#             recall_macro = recall_score(y, y_pred, average='macro')
#             f1_macro = f1_score(y, y_pred, average='macro')

#             # Get fire class recall specifically
#             recall_per_class = recall_score(y, y_pred, average=None)

#             # Prioritize fire class recall (assuming it's class 0 after encoding)
#             fire_class_idx = 0  # Adjust based on your label encoding
#             fire_recall = recall_per_class[fire_class_idx] if len(recall_per_class) > fire_class_idx else 0

#             # Combined safety score
#             safety_score = 0.4 * fire_recall + 0.3 * recall_macro + 0.3 * f1_macro
#             return safety_score

#         # Randomized search with cross-validation
#         rf_random = RandomizedSearchCV(
#             estimator=rf_base,
#             param_distributions=param_distributions,
#             n_iter=100,  # Increased iterations for better search
#             cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
#             scoring=fire_safety_scorer,
#             n_jobs=-1,
#             random_state=42,
#             verbose=1
#         )

#         # Fit the randomized search
#         rf_random.fit(X_train, y_train)

#         self.best_params = rf_random.best_params_
#         print(f"\nBest parameters found: {self.best_params}")
#         print(f"Best cross-validation score: {rf_random.best_score_:.4f}")

#         # Evaluate on validation set
#         best_model = rf_random.best_estimator_
#         val_score = best_model.score(X_val, y_val)
#         print(f"Validation set accuracy: {val_score:.4f}")

#         return rf_random.best_estimator_, rf_random

#     def train_model(self, X_train, y_train, X_val=None, y_val=None, tune_hyperparams=True):
#         """Train the Random Forest model with optional hyperparameter tuning"""
#         self.feature_names = X_train.columns.tolist()

#         if tune_hyperparams and X_val is not None and y_val is not None:
#             self.model, search_results = self.hyperparameter_tuning(X_train, y_train, X_val, y_val)
#             self.search_results = search_results
#         else:
#             # Use default optimized parameters
#             self.model = RandomForestClassifier(
#                 n_estimators=300,
#                 max_depth=20,
#                 min_samples_split=5,
#                 min_samples_leaf=2,
#                 max_features='sqrt',
#                 class_weight=self.class_weights,
#                 random_state=42,
#                 n_jobs=-1,
#                 oob_score=True,
#                 bootstrap=True
#             )
#             self.model.fit(X_train, y_train)

#         print(f"\nModel training completed!")
#         print(f"Out-of-bag score: {self.model.oob_score_:.4f}")

#         if hasattr(self, 'search_results'):
#             print(f"Best parameters used: {self.best_params}")

#     def plot_class_distribution(self, y_train_orig, y_train_balanced, y_val, y_test):
#         """Visualize class distribution across all sets"""
#         # Prepare data for plotting
#         train_orig_counts = y_train_orig.value_counts()
#         train_balanced_counts = y_train_balanced.value_counts()
#         val_counts = y_val.value_counts()
#         test_counts = y_test.value_counts()

#         fig = make_subplots(
#             rows=2, cols=2,
#             subplot_titles=('Original Training', 'Balanced Training', 'Validation', 'Test'),
#             specs=[[{"type": "bar"}, {"type": "bar"}],
#                    [{"type": "bar"}, {"type": "bar"}]]
#         )

#         # Original training
#         fig.add_trace(go.Bar(x=train_orig_counts.index, y=train_orig_counts.values,
#                            name='Original Training', marker_color='lightblue'), row=1, col=1)

#         # Balanced training
#         fig.add_trace(go.Bar(x=train_balanced_counts.index, y=train_balanced_counts.values,
#                            name='Balanced Training', marker_color='lightcoral'), row=1, col=2)

#         # Validation
#         fig.add_trace(go.Bar(x=val_counts.index, y=val_counts.values,
#                            name='Validation', marker_color='lightgreen'), row=2, col=1)

#         # Test
#         fig.add_trace(go.Bar(x=test_counts.index, y=test_counts.values,
#                            name='Test', marker_color='lightyellow'), row=2, col=2)

#         fig.update_layout(
#             title_text="Class Distribution Across Data Splits",
#             showlegend=False,
#             height=600
#         )

#         fig.show()

#     def plot_feature_importance(self, top_n=15):
#         """Visualize feature importance with interactive plot"""
#         if self.model is None:
#             print("Model not trained yet!")
#             return

#         # Get feature importances
#         importance_df = pd.DataFrame({
#             'feature': self.feature_names,
#             'importance': self.model.feature_importances_
#         }).sort_values('importance', ascending=True).tail(top_n)

#         # Create interactive bar plot
#         fig = go.Figure(go.Bar(
#             x=importance_df['importance'],
#             y=importance_df['feature'],
#             orientation='h',
#             marker_color='rgba(55, 128, 191, 0.7)',
#             marker_line_color='rgba(55, 128, 191, 1.0)',
#             marker_line_width=1
#         ))

#         fig.update_layout(
#             title=f'Top {top_n} Feature Importance (Random Forest)',
#             xaxis_title='Importance Score',
#             yaxis_title='Features',
#             height=600,
#             template='plotly_white'
#         )

#         fig.show()
#         return importance_df

#     def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
#         """Create interactive confusion matrix heatmap"""
#         cm = confusion_matrix(y_true, y_pred)

#         if class_names is None:
#             class_names = self.label_encoder.classes_

#         # Create annotations for the heatmap
#         annotations = []
#         for i in range(len(class_names)):
#             for j in range(len(class_names)):
#                 annotations.append(
#                     dict(
#                         x=j, y=i,
#                         text=str(cm[i, j]),
#                         showarrow=False,
#                         font=dict(color="white" if cm[i, j] > cm.max()/2 else "black", size=14)
#                     )
#                 )

#         fig = go.Figure(data=go.Heatmap(
#             z=cm,
#             x=class_names,
#             y=class_names,
#             colorscale='Blues',
#             showscale=True
#         ))

#         fig.update_layout(
#             title='Confusion Matrix',
#             xaxis_title='Predicted Label',
#             yaxis_title='True Label',
#             annotations=annotations,
#             height=500,
#             template='plotly_white'
#         )

#         fig.show()
#         return cm

#     def plot_learning_curves(self, X_train, y_train, cv=5):
#         """Generate and plot learning curves"""
#         print("Generating learning curves...")

#         train_sizes = np.linspace(0.1, 1.0, 10)

#         train_sizes_abs, train_scores, val_scores = learning_curve(
#             self.model, X_train, y_train,
#             train_sizes=train_sizes,
#             cv=cv,
#             n_jobs=-1,
#             scoring='f1_macro'
#         )

#         # Calculate means and standard deviations
#         train_mean = np.mean(train_scores, axis=1)
#         train_std = np.std(train_scores, axis=1)
#         val_mean = np.mean(val_scores, axis=1)
#         val_std = np.std(val_scores, axis=1)

#         fig = go.Figure()

#         # Training scores
#         fig.add_trace(go.Scatter(
#             x=train_sizes_abs, y=train_mean,
#             mode='lines+markers',
#             name='Training Score',
#             line=dict(color='blue'),
#             error_y=dict(type='data', array=train_std, visible=True)
#         ))

#         # Validation scores
#         fig.add_trace(go.Scatter(
#             x=train_sizes_abs, y=val_mean,
#             mode='lines+markers',
#             name='Cross-Validation Score',
#             line=dict(color='red'),
#             error_y=dict(type='data', array=val_std, visible=True)
#         ))

#         fig.update_layout(
#             title='Learning Curves (F1-Macro Score)',
#             xaxis_title='Training Set Size',
#             yaxis_title='F1-Macro Score',
#             template='plotly_white',
#             height=500
#         )

#         fig.show()

#     def plot_validation_curves(self, X_train, y_train, param_name='n_estimators',
#                              param_range=None, cv=5):
#         """Generate validation curves for hyperparameter analysis"""
#         if param_range is None:
#             if param_name == 'n_estimators':
#                 param_range = [50, 100, 200, 300, 500]
#             elif param_name == 'max_depth':
#                 param_range = [5, 10, 15, 20, 25, None]
#             else:
#                 param_range = [1, 2, 5, 10, 20]

#         print(f"Generating validation curves for {param_name}...")

#         train_scores, val_scores = validation_curve(
#             RandomForestClassifier(random_state=42, n_jobs=-1),
#             X_train, y_train,
#             param_name=param_name,
#             param_range=param_range,
#             cv=cv,
#             scoring='f1_macro',
#             n_jobs=-1
#         )

#         train_mean = np.mean(train_scores, axis=1)
#         train_std = np.std(train_scores, axis=1)
#         val_mean = np.mean(val_scores, axis=1)
#         val_std = np.std(val_scores, axis=1)

#         fig = go.Figure()

#         fig.add_trace(go.Scatter(
#             x=param_range, y=train_mean,
#             mode='lines+markers',
#             name='Training Score',
#             line=dict(color='blue'),
#             error_y=dict(type='data', array=train_std, visible=True)
#         ))

#         fig.add_trace(go.Scatter(
#             x=param_range, y=val_mean,
#             mode='lines+markers',
#             name='Validation Score',
#             line=dict(color='red'),
#             error_y=dict(type='data', array=val_std, visible=True)
#         ))

#         fig.update_layout(
#             title=f'Validation Curve - {param_name}',
#             xaxis_title=param_name,
#             yaxis_title='F1-Macro Score',
#             template='plotly_white',
#             height=500
#         )

#         fig.show()

#     def plot_roc_curves(self, X_test, y_test):
#         """Plot ROC curves for multiclass classification"""
#         y_proba = self.model.predict_proba(X_test)
#         n_classes = len(self.label_encoder.classes_)

#         fig = go.Figure()

#         # Plot ROC curve for each class
#         for i in range(n_classes):
#             y_test_binary = (y_test == i).astype(int)
#             fpr, tpr, _ = roc_curve(y_test_binary, y_proba[:, i])
#             auc_score = roc_auc_score(y_test_binary, y_proba[:, i])

#             fig.add_trace(go.Scatter(
#                 x=fpr, y=tpr,
#                 mode='lines',
#                 name=f'{self.label_encoder.classes_[i]} (AUC = {auc_score:.3f})',
#                 line=dict(width=2)
#             ))

#         # Add diagonal reference line
#         fig.add_trace(go.Scatter(
#             x=[0, 1], y=[0, 1],
#             mode='lines',
#             name='Random Classifier',
#             line=dict(dash='dash', color='black')
#         ))

#         fig.update_layout(
#             title='ROC Curves (One-vs-Rest)',
#             xaxis_title='False Positive Rate',
#             yaxis_title='True Positive Rate',
#             template='plotly_white',
#             height=500
#         )

#         fig.show()

#     def evaluate_model(self, X_test, y_test, plot_results=True):
#         """Comprehensive model evaluation with visualizations"""
#         print("=== SEEKLIYAB FIRE DETECTION SYSTEM EVALUATION ===\n")

#         # Predictions
#         y_pred = self.model.predict(X_test)
#         y_proba = self.model.predict_proba(X_test)

#         # Basic metrics
#         accuracy = accuracy_score(y_test, y_pred)
#         f1_macro = f1_score(y_test, y_pred, average='macro')
#         f1_weighted = f1_score(y_test, y_pred, average='weighted')

#         print(f"Overall Accuracy: {accuracy:.4f}")
#         print(f"F1-Score (Macro): {f1_macro:.4f}")
#         print(f"F1-Score (Weighted): {f1_weighted:.4f}\n")

#         # Detailed classification report
#         class_names = self.label_encoder.classes_
#         print("Classification Report:")
#         print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

#         # Store evaluation results
#         self.evaluation_results = {
#             'accuracy': accuracy,
#             'f1_macro': f1_macro,
#             'f1_weighted': f1_weighted,
#             'y_pred': y_pred,
#             'y_proba': y_proba,
#             'class_names': class_names
#         }

#         if plot_results:
#             # Plot confusion matrix
#             cm = self.plot_confusion_matrix(y_test, y_pred, class_names)

#             # Plot feature importance
#             importance_df = self.plot_feature_importance()

#             # Plot ROC curves
#             self.plot_roc_curves(X_test, y_test)

#         # Safety analysis
#         print("\n=== FIRE SAFETY ANALYSIS ===")
#         recall_scores = recall_score(y_test, y_pred, average=None)
#         precision_scores = precision_score(y_test, y_pred, average=None)

#         for i, class_name in enumerate(class_names):
#             print(f"{class_name}:")
#             print(f"  Recall: {recall_scores[i]:.4f}")
#             print(f"  Precision: {precision_scores[i]:.4f}")

#             if class_name == 'fire' and recall_scores[i] < 0.9:
#                 print(f"  WARNING: Fire recall below safety threshold!")

#         return self.evaluation_results

#     def demonstrate_tree_interpretability(self, X_train, max_trees=3, max_depth=3):
#         """
#         Demonstrate model interpretability by visualizing decision trees
#         """
#         print(f"\n=== MODEL INTERPRETABILITY DEMO ===")
#         print(f"Visualizing first {max_trees} decision trees from Random Forest")

#         if self.model is None:
#             print("Model not trained yet!")
#             return

#         for i in range(min(max_trees, len(self.model.estimators_))):
#             print(f"\nDecision Tree {i+1}:")

#             tree = self.model.estimators_[i]

#             # Export tree structure
#             dot_data = export_graphviz(
#                 tree,
#                 feature_names=X_train.columns,
#                 class_names=self.label_encoder.classes_,
#                 filled=True,
#                 max_depth=max_depth,
#                 impurity=False,
#                 proportion=True,
#                 rounded=True,
#                 special_characters=True
#             )

#             # Create and display graph
#             graph = graphviz.Source(dot_data)

#             # For Jupyter notebooks, this will display inline
#             try:
#                 display(graph)
#             except NameError:
#                 # If not in Jupyter, save to file
#                 graph.render(f'tree_{i+1}', format='png', cleanup=True)
#                 print(f"Tree {i+1} saved as tree_{i+1}.png")

#         return graph

#     def predict_fire_risk(self, sensor_data):
#         """
#         Real-time fire risk prediction for production deployment
#         """
#         if self.model is None:
#             raise ValueError("Model must be trained first!")

#         # Convert to DataFrame
#         df = pd.DataFrame([sensor_data])

#         # Apply feature engineering using training statistics
#         df_features = self.engineer_features(df, is_training=False)

#         # Select same features used in training
#         X = df_features[self.feature_names]

#         # Predict probabilities
#         probabilities = self.model.predict_proba(X)[0]
#         predicted_class_id = np.argmax(probabilities)
#         predicted_class = self.label_encoder.classes_[predicted_class_id]

#         # Calculate risk level based on probabilities
#         fire_prob = probabilities[0] if self.label_encoder.classes_[0] == 'fire' else 0
#         for i, class_name in enumerate(self.label_encoder.classes_):
#             if class_name == 'fire':
#                 fire_prob = probabilities[i]
#                 break

#         # Risk assessment
#         if fire_prob >= 0.7:
#             risk_level = "CRITICAL"
#         elif fire_prob >= 0.4:
#             risk_level = "HIGH"
#         elif fire_prob >= 0.2:
#             risk_level = "MEDIUM"
#         else:
#             risk_level = "LOW"

#         return {
#             'prediction': predicted_class,
#             'probabilities': {
#                 class_name: prob
#                 for class_name, prob in zip(self.label_encoder.classes_, probabilities)
#             },
#             'risk_level': risk_level,
#             'fire_probability': fire_prob,
#             'alert_required': risk_level in ['HIGH', 'CRITICAL']
#         }


# def main(apply_feature_engineering=False):
#     """
#     Main execution function demonstrating the complete SeekLiyab system
#     """
#     print("=== SEEKLIYAB FIRE DETECTION SYSTEM ===")
#     print("Production-ready model with proper data handling\n")

#     # Initialize detector with specified feature engineering setting
#     detector = SeekLiyabFireDetector(apply_feature_engineering=apply_feature_engineering)

#     # Define sensor column names (update these to match your Excel file columns)
#     SENSORS_COLUMN_NAMES = ['temperature_reading', 'air_quality_reading', 'carbon_monoxide_reading', 'smoke_reading']

#     # Step 1: Load data
#     RAW_DATA_FILE_PATH = '/content/sensor_readings_rows-06_12_25.xlsx'  # Update with your Excel file path
#     df = detector.load_data(RAW_DATA_FILE_PATH, SENSORS_COLUMN_NAMES)
#     if df is None:
#         print("Error: Could not load data. Please check the file path.")
#         return None

#     # Step 2: Split data FIRST to prevent data leakage
#     X_train, X_val, X_test, y_train, y_val, y_test = detector.split_data(df)

#     # Step 3: Learn feature engineering parameters from training data only (if enabled)
#     detector.fit_feature_engineering(X_train)

#     # Step 4: Apply feature engineering to all sets using training statistics
#     print("\nApplying feature engineering...")
#     X_train_eng = detector.engineer_features(X_train, is_training=True)
#     X_val_eng = detector.engineer_features(X_val, is_training=False)
#     X_test_eng = detector.engineer_features(X_test, is_training=False)

#     # Step 5: Handle class imbalance on training data only
#     print("\nHandling class imbalance...")
#     X_train_balanced, y_train_balanced = detector.handle_class_imbalance(
#         X_train_eng, y_train, strategy='smote_balanced'
#     )

#     # Step 6: Encode labels using training data mapping
#     print("\nEncoding labels...")
#     y_train_enc, y_val_enc, y_test_enc = detector.encode_labels(
#         y_train_balanced, y_val, y_test
#     )

#     # Step 7: Compute class weights
#     detector.compute_class_weights(y_train_enc)

#     # Step 8: Plot class distributions
#     print("\nVisualizing class distributions...")
#     detector.plot_class_distribution(y_train, y_train_balanced, y_val, y_test)

#     # Step 9: Train model with hyperparameter tuning
#     print("\nTraining model with hyperparameter tuning...")
#     detector.train_model(
#         X_train_balanced, y_train_enc,
#         X_val_eng, y_val_enc,
#         tune_hyperparams=True
#     )

#     # Step 10: Generate learning curves
#     print("\nGenerating learning curves...")
#     detector.plot_learning_curves(X_train_balanced, y_train_enc)

#     # Step 11: Generate validation curves for key parameters
#     print("\nGenerating validation curves...")
#     detector.plot_validation_curves(
#         X_train_balanced, y_train_enc,
#         param_name='n_estimators',
#         param_range=[50, 100, 200, 300, 500]
#     )

#     detector.plot_validation_curves(
#         X_train_balanced, y_train_enc,
#         param_name='max_depth',
#         param_range=[5, 10, 15, 20, 25, None]
#     )

#     # Step 12: Comprehensive model evaluation
#     print("\nEvaluating model...")
#     results = detector.evaluate_model(X_test_eng, y_test_enc, plot_results=True)

#     # Step 13: Demonstrate decision tree interpretability
#     print("\nDemonstrating model interpretability...")
#     detector.demonstrate_tree_interpretability(X_train_balanced, max_trees=3, max_depth=3)

#     # Step 14: Real-time prediction demo using actual data
#     print("\n=== REAL-TIME FIRE DETECTION DEMO USING ACTUAL DATA ===")

#     # Load the data again for prediction demonstration
#     print("Loading data for prediction demonstration...")
#     demo_df = detector.load_data(RAW_DATA_FILE_PATH, SENSORS_COLUMN_NAMES)

#     if demo_df is not None:
#         # # Sample a few random records from each class for demonstration
#         # demo_samples = []
#         # for label in demo_df['label'].unique():
#         #     class_samples = demo_df[demo_df['label'] == label].sample(min(2, len(demo_df[demo_df['label'] == label])), random_state=42)
#         #     demo_samples.append(class_samples)

#         demo_data = demo_df.reset_index(drop=True)

#         # Add prediction columns
#         predictions = []
#         risk_levels = []
#         fire_probabilities = []
#         alert_required_list = []

#         print(f"\nTesting {len(demo_data)} real samples from the dataset:")
#         print("="*80)

#         for idx, row in demo_data.iterrows():
#             # Prepare sensor data (exclude the label column)
#             sensor_data = {
#                 'temperature_reading': row['temperature_reading'],
#                 'air_quality_reading': row['air_quality_reading'],
#                 'carbon_monoxide_reading': row['carbon_monoxide_reading'],
#                 'smoke_reading': row['smoke_reading']
#             }

#             # Get prediction
#             result = detector.predict_fire_risk(sensor_data)

#             # Store results
#             predictions.append(result['prediction'])
#             risk_levels.append(result['risk_level'])
#             fire_probabilities.append(result['fire_probability'])
#             alert_required_list.append(result['alert_required'])

#             # Display results
#             print(f"\nSample {idx + 1} - Actual: {row['label']}")
#             print(f"Sensor Data: Temp={row['temperature_reading']:.1f}, Air={row['air_quality_reading']:.0f}, CO={row['carbon_monoxide_reading']:.0f}, Smoke={row['smoke_reading']:.0f}")
#             print(f"Prediction: {result['prediction']}")
#             print(f"Risk Level: {result['risk_level']}")
#             print(f"Fire Probability: {result['fire_probability']:.3f}")
#             print(f"Alert Required: {result['alert_required']}")

#             # Show all class probabilities
#             print("Class Probabilities:")
#             for class_name, prob in result['probabilities'].items():
#                 print(f"  {class_name}: {prob:.3f}")

#             # Check if prediction matches actual
#             prediction_correct = result['prediction'] == row['label']
#             print(f"Prediction Correct: {prediction_correct}")

#             if result['alert_required']:
#                 if result['risk_level'] == 'CRITICAL':
#                     print("🚨 ACTION: IMMEDIATE EVACUATION AND FIRE DEPARTMENT ALERT")
#                 else:
#                     print("⚠️  ACTION: INVESTIGATE AREA AND PREPARE SAFETY MEASURES")

#             print("-" * 80)

#         # Add predictions to the dataframe
#         demo_data_with_predictions = demo_data.copy()
#         demo_data_with_predictions['predicted_label'] = predictions
#         demo_data_with_predictions['risk_level'] = risk_levels
#         demo_data_with_predictions['fire_probability'] = fire_probabilities
#         demo_data_with_predictions['alert_required'] = alert_required_list
#         demo_data_with_predictions['prediction_correct'] = demo_data_with_predictions['label'] == demo_data_with_predictions['predicted_label']

#         # Summary statistics
#         accuracy_demo = (demo_data_with_predictions['prediction_correct'].sum() / len(demo_data_with_predictions)) * 100
#         print(f"\n=== DEMO PREDICTION SUMMARY ===")
#         print(f"Total samples tested: {len(demo_data_with_predictions)}")
#         print(f"Correct predictions: {demo_data_with_predictions['prediction_correct'].sum()}")
#         print(f"Demo accuracy: {accuracy_demo:.1f}%")
#         print(f"Alerts triggered: {sum(alert_required_list)}")

#         print("\nPrediction breakdown by actual class:")
#         for actual_class in demo_data_with_predictions['label'].unique():
#             class_data = demo_data_with_predictions[demo_data_with_predictions['label'] == actual_class]
#             class_accuracy = (class_data['prediction_correct'].sum() / len(class_data)) * 100
#             print(f"  {actual_class}: {class_accuracy:.1f}% accuracy ({class_data['prediction_correct'].sum()}/{len(class_data)})")

#         # Save results to CSV (optional)
#         try:
#             output_filename = 'seekliyab_predictions_demo.csv'
#             demo_data_with_predictions.to_csv(output_filename, index=False)
#             print(f"\n✅ Demo results saved to '{output_filename}'")
#         except Exception as e:
#             print(f"⚠️ Could not save demo results: {e}")

#         print(f"\nDemo DataFrame shape: {demo_data_with_predictions.shape}")
#         print("Columns:", list(demo_data_with_predictions.columns))

#     else:
#         print("Could not load data for demonstration")
#         # Fallback to manual test scenarios with updated column names
#         test_scenarios = [
#             {
#                 'name': 'Normal Office Environment',
#                 'data': {'temperature_reading': 24.5, 'air_quality_reading': 250, 'carbon_monoxide_reading': 350, 'smoke_reading': 180}
#             },
#             {
#                 'name': 'Potential Fire Warning',
#                 'data': {'temperature_reading': 45.0, 'air_quality_reading': 420, 'carbon_monoxide_reading': 520, 'smoke_reading': 320}
#             },
#             {
#                 'name': 'Fire Emergency Detected',
#                 'data': {'temperature_reading': 62.0, 'air_quality_reading': 780, 'carbon_monoxide_reading': 680, 'smoke_reading': 650}
#             },
#             {
#                 'name': 'Extreme Fire Conditions',
#                 'data': {'temperature_reading': 70.0, 'air_quality_reading': 950, 'carbon_monoxide_reading': 850, 'smoke_reading': 890}
#             }
#         ]

#         for scenario in test_scenarios:
#             print(f"\n--- {scenario['name']} ---")
#             result = detector.predict_fire_risk(scenario['data'])

#             print(f"Prediction: {result['prediction']}")
#             print(f"Risk Level: {result['risk_level']}")
#             print(f"Fire Probability: {result['fire_probability']:.3f}")
#             print(f"Alert Required: {result['alert_required']}")

#             print("All Class Probabilities:")
#             for class_name, prob in result['probabilities'].items():
#                 print(f"  {class_name}: {prob:.3f}")

#             if result['alert_required']:
#                 if result['risk_level'] == 'CRITICAL':
#                     print("ACTION: IMMEDIATE EVACUATION AND FIRE DEPARTMENT ALERT")
#                 else:
#                     print("ACTION: INVESTIGATE AREA AND PREPARE SAFETY MEASURES")

#     # Step 15: Feature importance analysis
#     print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
#     importance_df = pd.DataFrame({
#         'feature': detector.feature_names,
#         'importance': detector.model.feature_importances_
#     }).sort_values('importance', ascending=False)

#     print("\nTop 10 Most Important Features:")
#     for i, (idx, row) in enumerate(importance_df.head(10).iterrows()):
#         print(f"{i+1:2d}. {row['feature']:<25} {row['importance']:.6f}")

#     # Step 16: Model performance summary
#     print(f"\n=== FINAL MODEL PERFORMANCE SUMMARY ===")
#     print(f"Model Type: Random Forest with {detector.model.n_estimators} trees")
#     print(f"Features Used: {len(detector.feature_names)} (feature engineering: {'enabled' if detector.apply_feature_engineering else 'disabled'})")
#     print(f"Test Accuracy: {results['accuracy']:.4f}")
#     print(f"F1-Score (Macro): {results['f1_macro']:.4f}")
#     print(f"Out-of-Bag Score: {detector.model.oob_score_:.4f}")

#     if hasattr(detector, 'best_params'):
#         print(f"Optimized Hyperparameters: {detector.best_params}")

#     print(f"\n✅ SeekLiyab Fire Detection System ready for deployment!")
#     print(f"✅ Proper data splitting implemented (no data leakage)")
#     print(f"✅ Feature engineering: {'enabled' if detector.apply_feature_engineering else 'disabled'}")
#     print(f"✅ Class imbalance handled with SMOTE")
#     print(f"✅ Hyperparameter tuning completed")
#     print(f"✅ Comprehensive evaluation with visualizations")
#     print(f"✅ Model interpretability demonstrated")

#     return detector, results


# def analyze_hyperparameter_importance(detector):
#     """
#     Additional analysis of hyperparameter search results
#     """
#     if not hasattr(detector, 'search_results'):
#         print("No hyperparameter search results available")
#         return

#     # Get search results
#     search_results = detector.search_results
#     results_df = pd.DataFrame(search_results.cv_results_)

#     # Plot hyperparameter importance
#     params_to_analyze = ['n_estimators', 'max_depth', 'min_samples_split', 'max_features']

#     fig = make_subplots(
#         rows=2, cols=2,
#         subplot_titles=params_to_analyze
#     )

#     for i, param in enumerate(params_to_analyze):
#         row = (i // 2) + 1
#         col = (i % 2) + 1

#         param_col = f'param_{param}'
#         if param_col in results_df.columns:
#             # Group by parameter value and get mean score
#             param_scores = results_df.groupby(param_col)['mean_test_score'].mean().reset_index()

#             fig.add_trace(
#                 go.Scatter(
#                     x=param_scores[param_col],
#                     y=param_scores['mean_test_score'],
#                     mode='markers+lines',
#                     name=param
#                 ),
#                 row=row, col=col
#             )

#     fig.update_layout(
#         title="Hyperparameter Impact on Performance",
#         height=600,
#         showlegend=False
#     )

#     fig.show()


# def example_with_feature_engineering():
#     """
#     Example of using the detector with feature engineering enabled
#     """
#     print("=== EXAMPLE: SEEKLIYAB WITH FEATURE ENGINEERING ===")

#     # Initialize detector with feature engineering enabled
#     detector = SeekLiyabFireDetector(apply_feature_engineering=True)

#     # Define sensor column names (update these to match your Excel file columns)
#     SENSORS_COLUMN_NAMES = ['temperature_reading', 'air_quality_reading', 'carbon_monoxide_reading', 'smoke_reading']

#     # Load and process data
#     RAW_DATA_FILE_PATH = '/content/sensor_readings_rows-06_12_25.xlsx'
#     df = detector.load_data(RAW_DATA_FILE_PATH, SENSORS_COLUMN_NAMES)

#     if df is not None:
#         # Continue with the full pipeline...
#         X_train, X_val, X_test, y_train, y_val, y_test = detector.split_data(df)
#         detector.fit_feature_engineering(X_train)

#         print("Feature engineering will be applied during training")
#         print("This will create additional features like:")
#         print("- Temperature z-scores and thresholds")
#         print("- Gas concentration z-scores")
#         print("- Interaction features")
#         print("- Fire risk composite scores")
#         print("- Anomaly indicators")

#     return detector


# if __name__ == "__main__":
#     # Execute the main pipeline with feature engineering disabled (default)
#     detector, results = main(apply_feature_engineering=False)

#     # Additional hyperparameter analysis
#     if detector is not None:
#         print("\n=== HYPERPARAMETER SEARCH ANALYSIS ===")
#         analyze_hyperparameter_importance(detector)

#         # Save model for future use (optional)
#         try:
#             import joblib
#             joblib.dump(detector, 'seekliyab_fire_detector_model.pkl')
#             print("\n✅ Model saved as 'seekliyab_fire_detector_model.pkl'")
#         except ImportError:
#             print("\n⚠️ joblib not available - model not saved")

#     # Example with feature engineering enabled
#     print("\n" + "="*60)
#     print("Running example with feature engineering enabled...")
#     example_detector = main(apply_feature_engineering=True)

#     print(f"\n🔥 SEEKLIYAB FIRE DETECTION SYSTEM DEPLOYMENT READY! 🔥")