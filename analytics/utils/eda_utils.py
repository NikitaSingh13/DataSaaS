import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
from django.conf import settings
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for server environments
plt.switch_backend('Agg')

def read_file(file_path):
    """
    Read CSV or Excel file into pandas DataFrame
    """
    try:
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return df
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

def get_summary(df):
    """
    Generate comprehensive summary statistics (JSON-serializable)
    """
    summary = {
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        'missing_values': [],
        'numerical_summary': [],
        'categorical_summary': []
    }

    # Missing values
    missing_data = df.isnull().sum()
    for col in missing_data.index:
        if missing_data[col] > 0:
            summary['missing_values'].append({
                'column': str(col),
                'count': int(missing_data[col]),
                'percentage': float((missing_data[col] / len(df)) * 100)
            })

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        describe_df = df[numeric_cols].describe()
        for col in numeric_cols:
            summary['numerical_summary'].append({
                'column': str(col),
                'count': int(describe_df.loc['count', col]),
                'mean': float(describe_df.loc['mean', col]),
                'std': float(describe_df.loc['std', col]),
                'min': float(describe_df.loc['min', col]),
                'max': float(describe_df.loc['max', col])
            })

    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        col_series = df[col].astype(str)
        value_counts = col_series.value_counts()
        summary['categorical_summary'].append({
            'column': str(col),
            'unique_count': int(df[col].nunique()),
            'top_value': str(value_counts.index[0]) if len(value_counts) > 0 else 'N/A'
        })

    return summary


def generate_plots(df, file_id):
    """
    Generate EDA plots and save them to media folder
    """
    # Create plots directory
    plots_dir = os.path.join(settings.MEDIA_ROOT, 'plots', str(file_id))
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_paths = {}
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    try:
        # 1. Missing values heatmap
        if df.isnull().sum().sum() > 0:
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.isnull(), cmap='viridis', cbar=True)
            plt.title('Missing Values Heatmap')
            plt.tight_layout()
            missing_path = os.path.join(plots_dir, 'missing_values.png')
            plt.savefig(missing_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['missing_values'] = f'plots/{file_id}/missing_values.png'
        
        # 2. Histograms for numeric columns
        if numeric_cols:
            n_cols = min(4, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 4 * n_rows))
            for i, col in enumerate(numeric_cols[:12]):  # Limit to 12 columns
                plt.subplot(n_rows, n_cols, i + 1)
                df[col].hist(bins=30, alpha=0.7)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            hist_path = os.path.join(plots_dir, 'histograms.png')
            plt.savefig(hist_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['histograms'] = f'plots/{file_id}/histograms.png'
        
        # 3. Boxplots for numeric columns
        if numeric_cols:
            n_cols = min(4, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 4 * n_rows))
            for i, col in enumerate(numeric_cols[:12]):  # Limit to 12 columns
                plt.subplot(n_rows, n_cols, i + 1)
                df[col].dropna().plot(kind='box')
                plt.title(f'Boxplot of {col}')
                plt.ylabel(col)
            
            plt.tight_layout()
            box_path = os.path.join(plots_dir, 'boxplots.png')
            plt.savefig(box_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['boxplots'] = f'plots/{file_id}/boxplots.png'
        
        # 4. Correlation heatmap
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            corr_path = os.path.join(plots_dir, 'correlation_heatmap.png')
            plt.savefig(corr_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['correlation_heatmap'] = f'plots/{file_id}/correlation_heatmap.png'
        
        # 5. Categorical plots (top categories)
        for col in categorical_cols[:5]:  # Limit to 5 categorical columns
            try:
                # Convert to string to handle any object types
                col_data = df[col].astype(str)
                
                if col_data.nunique() <= 20:  # Only plot if reasonable number of categories
                    plt.figure(figsize=(10, 6))
                    value_counts = col_data.value_counts().head(10)
                    value_counts.plot(kind='bar')
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Count')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Clean column name for filename
                    clean_col = "".join(c for c in str(col) if c.isalnum() or c in (' ', '_')).replace(' ', '_')
                    cat_path = os.path.join(plots_dir, f'categorical_{clean_col}.png')
                    plt.savefig(cat_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    plot_paths[f'categorical_{clean_col}'] = f'plots/{file_id}/categorical_{clean_col}.png'
            except Exception as e:
                print(f"Error plotting categorical column {col}: {str(e)}")
                continue
        
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
    
    return plot_paths
