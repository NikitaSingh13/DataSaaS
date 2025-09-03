from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from accounts.models import UploadedFile, Insight
from .utils.eda_utils import read_file, get_summary, generate_plots
import json
import os

@login_required
def eda_view(request, file_id):
    """
    Perform EDA on uploaded file and display results
    """
    # Get the uploaded file
    uploaded_file = get_object_or_404(UploadedFile, id=file_id, user=request.user)
    
    try:
        # Read the file into DataFrame
        file_path = uploaded_file.file.path
        df = read_file(file_path)
        
        # Generate summary statistics
        summary = get_summary(df)
        
        # Generate plots
        plot_paths = generate_plots(df, file_id)
        
        # Save insight to database
        insight_data = {
            'summary': summary,
            'plots': plot_paths
        }
        
        # Create or update insight
        insight, created = Insight.objects.get_or_create(
            uploaded_file=uploaded_file,
            defaults={'insight_data': insight_data}
        )
        
        if not created:
            insight.insight_data = insight_data
            insight.save()
        
        # Prepare context for template
        context = {
            'uploaded_file': uploaded_file,
            'summary': summary,
            'plots': [{'name': name.replace('_', ' ').title(), 'path': path} for name, path in plot_paths.items()],
            'df_head': df.head().to_html(classes='table table-striped table-sm', table_id='data-preview'),
            'insight': insight
        }
        
        return render(request, 'analytics/eda_simple.html', context)
        
    except Exception as e:
        messages.error(request, f'Error performing EDA: {str(e)}')
        return redirect('accounts:dashboard')

@login_required
def column_analysis_view(request, file_id):
    """
    Detailed analysis of individual columns
    """
    uploaded_file = get_object_or_404(UploadedFile, id=file_id, user=request.user)
    
    try:
        # Read the file
        file_path = uploaded_file.file.path
        df = read_file(file_path)
        
        # Get column information
        columns_info = []
        for col in df.columns:
            non_null_count = len(df) - int(df[col].isnull().sum())
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'non_null_count': non_null_count,
                'null_percentage': round((df[col].isnull().sum() / len(df)) * 100, 2),
                'unique_count': int(df[col].nunique()),
                'memory_usage': int(df[col].memory_usage(deep=True))
            }
            
            # Add specific stats based on data type
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'mean': round(df[col].mean(), 2) if not df[col].isnull().all() else None,
                    'median': round(df[col].median(), 2) if not df[col].isnull().all() else None,
                    'std': round(df[col].std(), 2) if not df[col].isnull().all() else None,
                    'min': round(df[col].min(), 2) if not df[col].isnull().all() else None,
                    'max': round(df[col].max(), 2) if not df[col].isnull().all() else None,
                })
            else:
                # For categorical columns
                if not df[col].isnull().all():
                    col_info['top_values'] = df[col].value_counts().head(5).to_dict()
            
            columns_info.append(col_info)
        
        context = {
            'uploaded_file': uploaded_file,
            'columns_info': columns_info,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        return render(request, 'analytics/column_analysis_simple.html', context)
        
    except Exception as e:
        messages.error(request, f'Error analyzing columns: {str(e)}')
        return redirect('accounts:dashboard')
