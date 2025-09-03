"""
ML Training Utilities for DataSaaS
Handles preprocessing, model training, and evaluation
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path
from django.conf import settings
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, 
    confusion_matrix, classification_report
)
from sklearn.dummy import DummyRegressor, DummyClassifier
import warnings
warnings.filterwarnings('ignore')


def detect_problem_type(df, target_column, threshold=20):
    """
    Detect if the problem is regression or classification
    
    Args:
        df: pandas DataFrame
        target_column: string, name of target column
        threshold: int, max unique values for classification
    
    Returns:
        tuple: (problem_type, target_info)
    """
    try:
        target_series = df[target_column].dropna()
        unique_count = target_series.nunique()
        
        # Check if target is numeric
        is_numeric = pd.api.types.is_numeric_dtype(target_series)
        
        target_info = {
            'unique_count': int(unique_count),
            'is_numeric': bool(is_numeric),
            'data_type': str(target_series.dtype),
            'sample_values': [val.item() if hasattr(val, 'item') else val for val in target_series.head(10).tolist()],
            'null_count': int(df[target_column].isnull().sum())
        }
        
        # Decision logic
        if is_numeric and unique_count > threshold:
            problem_type = 'regression'
        else:
            problem_type = 'classification'
            
        return problem_type, target_info
        
    except Exception as e:
        raise Exception(f"Error detecting problem type: {str(e)}")


def preprocess_data(df, target_column, problem_type):
    """
    Comprehensive data preprocessing pipeline
    
    Args:
        df: pandas DataFrame
        target_column: string, name of target column
        problem_type: string, 'regression' or 'classification'
    
    Returns:
        dict: preprocessed data and metadata
    """
    try:
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Initial validation
        if df_clean.empty:
            raise ValueError("Input DataFrame is empty")
        
        if target_column not in df_clean.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Separate features and target
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column].copy()
        
        print(f"Initial data shape: X={X.shape}, y={y.shape}")
        print(f"Initial feature columns: {list(X.columns)}")
        print(f"Target column '{target_column}' unique values: {y.nunique()}")
        
        preprocessing_steps = []
        
        # 1. Remove constant columns (all same value) - but be more conservative
        constant_cols = []
        for col in X.columns:
            # Only remove if truly constant (not just low variance)
            unique_vals = X[col].dropna().nunique()
            if unique_vals <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            X = X.drop(columns=constant_cols)
            preprocessing_steps.append(f"Removed {len(constant_cols)} constant columns: {constant_cols}")
            print(f"After removing constant columns: X={X.shape}")
        
        # 2. Remove ID-like columns - be much more conservative
        id_like_cols = []
        for col in X.columns:
            # Only remove if it's really obviously an ID column
            unique_ratio = X[col].nunique() / len(X)
            # Must have both high uniqueness AND many values AND likely string identifiers
            if (unique_ratio > 0.95 and 
                X[col].nunique() > 500 and 
                X[col].dtype == 'object' and
                str(col).lower() in ['id', 'identifier', 'key', 'uuid', 'guid']):
                id_like_cols.append(col)
        
        if id_like_cols:
            X = X.drop(columns=id_like_cols)
            preprocessing_steps.append(f"Removed {len(id_like_cols)} ID-like columns: {id_like_cols}")
            print(f"After removing ID-like columns: X={X.shape}")
        else:
            print(f"No ID-like columns removed. X shape remains: {X.shape}")
        
        # 3. Handle missing values in target
        target_missing = y.isnull().sum()
        if target_missing > 0:
            y = y.dropna()
            X = X.loc[y.index]  # Keep corresponding rows
            preprocessing_steps.append(f"Removed {target_missing} rows with missing target values")
            print(f"After removing missing targets: X={X.shape}, y={y.shape}")
        
        # Safety check: ensure we have at least some features
        if X.shape[1] == 0:
            # If all features were removed, let's be more conservative
            print("WARNING: All features were removed. Restoring original features...")
            X = df_clean.drop(columns=[target_column])
            preprocessing_steps.append("WARNING: Restored all original features due to over-aggressive cleaning")
            
            # Only remove truly problematic columns
            problem_cols = []
            for col in X.columns:
                # Only remove columns that are literally constant (all exactly the same value)
                if X[col].nunique() == 1:
                    problem_cols.append(col)
            
            if problem_cols:
                X = X.drop(columns=problem_cols)
                preprocessing_steps.append(f"Removed only truly constant columns: {problem_cols}")
            
            print(f"After conservative cleaning: X={X.shape}")
        
        # Validation: Check if we still have data
        if X.shape[0] == 0:
            raise ValueError("No data remaining after removing missing target values")
        
        if X.shape[1] == 0:
            raise ValueError("No features remaining after preprocessing - this shouldn't happen with safety checks")
        
        # 4. Handle missing values in features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"Numeric columns: {len(numeric_cols)}, Categorical columns: {len(categorical_cols)}")
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            missing_count = X[col].isnull().sum()
            if missing_count > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                preprocessing_steps.append(f"Filled {missing_count} missing values in '{col}' with median: {median_val:.2f}")
        
        # Fill categorical missing values with mode or 'Unknown'
        for col in categorical_cols:
            missing_count = X[col].isnull().sum()
            if missing_count > 0:
                mode_val = X[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                X[col] = X[col].fillna(fill_val)
                preprocessing_steps.append(f"Filled {missing_count} missing values in '{col}' with mode: '{fill_val}'")
        
        # 5. Encode categorical variables - with intelligent handling of high cardinality
        encoded_cols = []
        if categorical_cols:
            print(f"Processing {len(categorical_cols)} categorical columns...")
            
            # Check cardinality of each categorical column
            low_cardinality_cols = []
            high_cardinality_cols = []
            
            for col in categorical_cols:
                unique_count = X[col].nunique()
                print(f"Column '{col}': {unique_count} unique values")
                
                if unique_count <= 20:  # Low cardinality - safe for one-hot encoding
                    low_cardinality_cols.append(col)
                else:  # High cardinality - use different strategy
                    high_cardinality_cols.append(col)
            
            # One-hot encode low cardinality columns
            if low_cardinality_cols:
                X_encoded = pd.get_dummies(X, columns=low_cardinality_cols, drop_first=True)
                encoded_cols = [col for col in X_encoded.columns if col not in X.columns]
                X = X_encoded
                preprocessing_steps.append(f"One-hot encoded {len(low_cardinality_cols)} low-cardinality columns: {low_cardinality_cols}")
                print(f"After one-hot encoding low cardinality: X={X.shape}")
            
            # Handle high cardinality columns differently
            if high_cardinality_cols:
                for col in high_cardinality_cols:
                    # Strategy 1: Keep only top N most frequent categories, rest as 'Other'
                    top_n = 10  # Keep only top 10 categories
                    top_categories = X[col].value_counts().head(top_n).index.tolist()
                    
                    # Create new column with reduced categories
                    X[col + '_grouped'] = X[col].apply(
                        lambda x: x if x in top_categories else 'Other'
                    )
                    
                    # One-hot encode the grouped version
                    grouped_encoded = pd.get_dummies(X[col + '_grouped'], prefix=col, drop_first=True)
                    X = pd.concat([X, grouped_encoded], axis=1)
                    
                    # Remove original columns
                    X = X.drop(columns=[col, col + '_grouped'])
                    
                    encoded_cols.extend(grouped_encoded.columns.tolist())
                
                preprocessing_steps.append(f"Grouped {len(high_cardinality_cols)} high-cardinality columns to top-{top_n} + 'Other': {high_cardinality_cols}")
                print(f"After handling high cardinality columns: X={X.shape}")
        
        print(f"Total features after encoding: {X.shape[1]}")
        
        # Safety check for memory usage
        estimated_memory_gb = (X.shape[0] * X.shape[1] * 8) / (1024**3)  # 8 bytes per float64
        print(f"Estimated memory usage: {estimated_memory_gb:.2f} GB")
        
        if estimated_memory_gb > 4:  # If more than 4GB
            print("WARNING: High memory usage detected. Reducing features...")
            # Further reduce features if still too large
            if X.shape[1] > 1000:
                # Keep only numeric columns and a subset of encoded columns
                numeric_feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                encoded_feature_cols = [col for col in X.columns if col not in numeric_feature_cols]
                
                # Keep all numeric + first 500 encoded features
                features_to_keep = numeric_feature_cols + encoded_feature_cols[:500]
                X = X[features_to_keep]
                preprocessing_steps.append(f"Reduced features due to memory constraints: kept {len(features_to_keep)} out of {len(numeric_feature_cols) + len(encoded_feature_cols)} features")
                print(f"After feature reduction: X={X.shape}")
        
        # 6. Handle target variable for classification
        label_encoder = None
        original_classes = None
        if problem_type == 'classification':
            if not pd.api.types.is_numeric_dtype(y):
                label_encoder = LabelEncoder()
                original_classes = y.unique().tolist()
                y = label_encoder.fit_transform(y)
                preprocessing_steps.append(f"Label encoded target variable. Classes: {original_classes}")
        
        # Final data info
        final_info = {
            'feature_count': int(X.shape[1]),
            'sample_count': int(X.shape[0]),
            'numeric_features': int(len(X.select_dtypes(include=[np.number]).columns)),
            'categorical_features_encoded': int(len(encoded_cols)),
            'target_classes': int(len(np.unique(y))) if problem_type == 'classification' else None
        }
        
        # Final validation
        print(f"Final data shape: X={X.shape}, y={y.shape}")
        
        if X.shape[0] == 0:
            raise ValueError("No samples remaining after preprocessing")
        
        if X.shape[1] == 0:
            raise ValueError("No features remaining after preprocessing")
        
        if len(y) == 0:
            raise ValueError("No target values remaining after preprocessing")
        
        # Ensure all features are numeric
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            print(f"Converting non-numeric columns to numeric: {list(non_numeric_cols)}")
            for col in non_numeric_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                # Fill any NaN values created from failed conversions
                if X[col].isnull().any():
                    X[col] = X[col].fillna(0)
        
        # Check for any remaining NaN values
        if X.isnull().any().any():
            print("WARNING: Found remaining NaN values in features, filling with 0")
            X = X.fillna(0)
        
        if pd.isna(y).any():
            raise ValueError("NaN values found in target after preprocessing")
        
        # Convert to proper numpy arrays if needed
        if isinstance(y, pd.Series):
            y = y.values
        
        print(f"Final data types - Features: {X.dtypes.unique()}, Target: {type(y[0])}")
        
        return {
            'X': X,
            'y': y,
            'feature_names': X.columns.tolist(),
            'preprocessing_steps': preprocessing_steps,
            'removed_columns': constant_cols + id_like_cols,
            'label_encoder': label_encoder,
            'original_classes': original_classes,
            'final_info': final_info
        }
        
    except Exception as e:
        raise Exception(f"Error in preprocessing: {str(e)}")


def train_models(X, y, problem_type, test_size=0.2, random_state=42):
    """
    Train baseline and advanced models
    
    Args:
        X: pandas DataFrame, features
        y: pandas Series/array, target
        problem_type: string, 'regression' or 'classification'
        test_size: float, proportion for test set
        random_state: int, for reproducibility
    
    Returns:
        dict: trained models and evaluation results
    """
    try:
        # Initial validation
        print(f"Training models with X shape: {X.shape}, y shape: {y.shape if hasattr(y, 'shape') else len(y)}")
        
        if X is None or (hasattr(X, 'empty') and X.empty):
            raise ValueError("Features (X) is None or empty")
        
        if y is None or len(y) == 0:
            raise ValueError("Target (y) is None or empty")
        
        if X.shape[0] != len(y):
            raise ValueError(f"Mismatch between features ({X.shape[0]}) and target ({len(y)}) sample counts")
        
        if X.shape[0] < 2:
            raise ValueError(f"Insufficient samples for training: {X.shape[0]}. Need at least 2 samples.")
        
        # Check for infinite or very large values - only on numeric data
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_data = X[numeric_cols]
            if np.isinf(numeric_data.values).any():
                print("WARNING: Infinite values found in numeric features, replacing with NaN")
                X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
                # Fill NaN values created from inf replacement
                for col in numeric_cols:
                    if X[col].isnull().any():
                        median_val = X[col].median()
                        X[col] = X[col].fillna(median_val)
        
        # Check target for infinite values
        if pd.api.types.is_numeric_dtype(y):
            if np.isinf(y).any():
                print("WARNING: Infinite values found in target, replacing with median")
                median_target = pd.Series(y).median()
                y = pd.Series(y).replace([np.inf, -np.inf], median_target).values
        
        # Final check: ensure all features are numeric
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            print(f"WARNING: Non-numeric columns found: {list(non_numeric_cols)}")
            # Try to convert to numeric, coercing errors to NaN
            for col in non_numeric_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                # Fill any NaN values created
                if X[col].isnull().any():
                    median_val = X[col].median()
                    if pd.isna(median_val):  # If all values were non-numeric
                        X[col] = 0  # Fill with 0 as last resort
                    else:
                        X[col] = X[col].fillna(median_val)
        
        print(f"Final validation - X data types: {X.dtypes.value_counts().to_dict()}")
        print(f"Final validation - y data type: {type(y[0]) if hasattr(y, '__iter__') and len(y) > 0 else type(y)}")
        
        # Train/test split
        if problem_type == 'classification' and len(np.unique(y)) > 1:
            # Stratified split for classification
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(splitter.split(X, y))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        else:
            # Regular split for regression or single class
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        models = {}
        results = {}
        
        if problem_type == 'regression':
            # Baseline: Dummy regressor (mean)
            dummy = DummyRegressor(strategy='mean')
            dummy.fit(X_train, y_train)
            models['baseline'] = dummy
            
            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            models['linear_regression'] = lr
            
            # Random Forest Regressor
            rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
            rf.fit(X_train, y_train)
            models['random_forest'] = rf
            
            # Evaluate all models
            for name, model in models.items():
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'rmse': rmse,
                    'r2': r2,
                    'predictions': y_pred,
                    'model': model
                }
        
        else:  # classification
            # Baseline: Dummy classifier (most frequent)
            dummy = DummyClassifier(strategy='most_frequent')
            dummy.fit(X_train, y_train)
            models['baseline'] = dummy
            
            # Logistic Regression
            lr = LogisticRegression(random_state=random_state, max_iter=1000)
            lr.fit(X_train, y_train)
            models['logistic_regression'] = lr
            
            # Random Forest Classifier
            rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
            rf.fit(X_train, y_train)
            models['random_forest'] = rf
            
            # Evaluate all models
            for name, model in models.items():
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                results[name] = {
                    'accuracy': accuracy,
                    'confusion_matrix': conf_matrix.tolist(),
                    'predictions': y_pred,
                    'model': model
                }
        
        return {
            'models': models,
            'results': results,
            'test_data': {
                'X_test': X_test,
                'y_test': y_test,
                'y_train': y_train
            },
            'split_info': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'test_ratio': test_size
            }
        }
        
    except Exception as e:
        raise Exception(f"Error training models: {str(e)}")


def generate_ml_plots(results, problem_type, file_id, feature_names=None):
    """
    Generate visualization plots for ML results
    
    Args:
        results: dict, model results from train_models
        problem_type: string, 'regression' or 'classification'
        file_id: int, for organizing plots
        feature_names: list, for feature importance
    
    Returns:
        dict: plot file paths
    """
    try:
        # Create ML plots directory
        plots_dir = os.path.join(settings.MEDIA_ROOT, 'ml_plots', str(file_id))
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_paths = {}
        
        if problem_type == 'regression':
            # 1. Model comparison plot
            plt.figure(figsize=(12, 6))
            models = ['baseline', 'linear_regression', 'random_forest']
            rmse_scores = [results['results'][model]['rmse'] for model in models]
            r2_scores = [results['results'][model]['r2'] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            plt.subplot(1, 2, 1)
            plt.bar(x, rmse_scores, width, label='RMSE', color=['red', 'orange', 'green'])
            plt.xlabel('Models')
            plt.ylabel('RMSE')
            plt.title('Model Comparison - RMSE')
            plt.xticks(x, [m.replace('_', ' ').title() for m in models])
            
            plt.subplot(1, 2, 2)
            plt.bar(x, r2_scores, width, label='R²', color=['red', 'orange', 'green'])
            plt.xlabel('Models')
            plt.ylabel('R² Score')
            plt.title('Model Comparison - R²')
            plt.xticks(x, [m.replace('_', ' ').title() for m in models])
            
            plt.tight_layout()
            comparison_path = os.path.join(plots_dir, 'model_comparison.png')
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['model_comparison'] = f'ml_plots/{file_id}/model_comparison.png'
            
            # 2. Predictions vs Actual (best model)
            best_model = min(models, key=lambda m: results['results'][m]['rmse'])
            y_test = results['test_data']['y_test']
            y_pred = results['results'][best_model]['predictions']
            
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Predictions vs Actual - {best_model.replace("_", " ").title()}')
            plt.tight_layout()
            
            pred_path = os.path.join(plots_dir, 'predictions_vs_actual.png')
            plt.savefig(pred_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['predictions_vs_actual'] = f'ml_plots/{file_id}/predictions_vs_actual.png'
            
        else:  # classification
            # 1. Model accuracy comparison
            plt.figure(figsize=(10, 6))
            models = ['baseline', 'logistic_regression', 'random_forest']
            accuracies = [results['results'][model]['accuracy'] for model in models]
            
            colors = ['red', 'orange', 'green']
            bars = plt.bar(models, accuracies, color=colors)
            plt.xlabel('Models')
            plt.ylabel('Accuracy')
            plt.title('Model Comparison - Accuracy')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            accuracy_path = os.path.join(plots_dir, 'accuracy_comparison.png')
            plt.savefig(accuracy_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['accuracy_comparison'] = f'ml_plots/{file_id}/accuracy_comparison.png'
            
            # 2. Confusion matrix (best model)
            best_model = max(models, key=lambda m: results['results'][m]['accuracy'])
            conf_matrix = np.array(results['results'][best_model]['confusion_matrix'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - {best_model.replace("_", " ").title()}')
            plt.tight_layout()
            
            conf_path = os.path.join(plots_dir, 'confusion_matrix.png')
            plt.savefig(conf_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['confusion_matrix'] = f'ml_plots/{file_id}/confusion_matrix.png'
        
        # 3. Feature importance (for tree-based models)
        if 'random_forest' in results['models'] and feature_names:
            rf_model = results['models']['random_forest']
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1]
                top_features = min(15, len(feature_names))  # Show top 15 features
                
                plt.figure(figsize=(10, 8))
                plt.bar(range(top_features), importances[indices[:top_features]])
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.title('Feature Importance - Random Forest')
                plt.xticks(range(top_features), 
                          [feature_names[i] for i in indices[:top_features]], 
                          rotation=45, ha='right')
                plt.tight_layout()
                
                importance_path = os.path.join(plots_dir, 'feature_importance.png')
                plt.savefig(importance_path, dpi=150, bbox_inches='tight')
                plt.close()
                plot_paths['feature_importance'] = f'ml_plots/{file_id}/feature_importance.png'
        
        return plot_paths
        
    except Exception as e:
        print(f"Error generating ML plots: {str(e)}")
        return {}


def save_model_artifacts(models, preprocessing_info, results, file_id, target_column):
    """
    Save trained models and metadata for reproducibility
    
    Args:
        models: dict, trained models
        preprocessing_info: dict, preprocessing metadata
        results: dict, training results
        file_id: int, for organizing files
        target_column: string, name of target column
    
    Returns:
        dict: saved file paths
    """
    try:
        # Create models directory
        models_dir = os.path.join(settings.MEDIA_ROOT, 'models', str(file_id))
        os.makedirs(models_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save each model
        for model_name, model in models.items():
            model_path = os.path.join(models_dir, f'{model_name}.joblib')
            joblib.dump(model, model_path)
            saved_files[f'{model_name}_model'] = f'models/{file_id}/{model_name}.joblib'
        
        # Save preprocessing metadata
        metadata = {
            'target_column': target_column,
            'preprocessing_steps': preprocessing_info['preprocessing_steps'],
            'feature_names': preprocessing_info['feature_names'],
            'removed_columns': preprocessing_info['removed_columns'],
            'final_info': preprocessing_info['final_info'],
            'label_encoder_classes': preprocessing_info.get('original_classes'),
            'model_results': {
                name: {k: v for k, v in result.items() if k != 'model'}
                for name, result in results['results'].items()
            }
        }
        
        metadata_path = os.path.join(models_dir, 'training_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        saved_files['metadata'] = f'models/{file_id}/training_metadata.joblib'
        
        # Save label encoder if exists
        if preprocessing_info.get('label_encoder'):
            encoder_path = os.path.join(models_dir, 'label_encoder.joblib')
            joblib.dump(preprocessing_info['label_encoder'], encoder_path)
            saved_files['label_encoder'] = f'models/{file_id}/label_encoder.joblib'
        
        return saved_files
        
    except Exception as e:
        raise Exception(f"Error saving model artifacts: {str(e)}")
