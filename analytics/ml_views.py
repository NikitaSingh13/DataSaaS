"""
ML Training Views for DataSaaS
Handles target column selection, model training, and results display
"""

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.utils import timezone
from django.core.management import call_command
from django.conf import settings
import json
import threading
import pandas as pd
import numpy as np

from accounts.models import UploadedFile, MLTrainingJob
from .utils.eda_utils import read_file
from .utils.ml_utils import (
    detect_problem_type, preprocess_data, train_models, 
    generate_ml_plots, save_model_artifacts
)


def convert_to_serializable(obj):
    """
    Convert NumPy/Pandas types to JSON serializable Python types
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


@login_required
def select_target_column(request, file_id):
    """
    View to display target column selection form
    Shows dataset columns and lets user pick target for ML training
    """
    try:
        # Get the uploaded file
        uploaded_file = get_object_or_404(UploadedFile, id=file_id, user=request.user)
        
        # Read the dataset to get column information
        df = read_file(uploaded_file.file.path)
        
        # Get column information for the form
        columns_info = []
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'unique_count': int(df[col].nunique()),
                'null_count': int(df[col].isnull().sum()),
                'sample_values': df[col].dropna().head(5).tolist(),
                'is_numeric': pd.api.types.is_numeric_dtype(df[col])
            }
            columns_info.append(col_info)
        
        # Handle form submission
        if request.method == 'POST':
            target_column = request.POST.get('target_column')
            
            if not target_column or target_column not in df.columns:
                messages.error(request, 'Please select a valid target column.')
                return render(request, 'analytics/select_target.html', {
                    'uploaded_file': uploaded_file,
                    'columns_info': columns_info,
                    'dataset_shape': df.shape
                })
            
            # Detect problem type
            try:
                problem_type, target_info = detect_problem_type(df, target_column)
                
                # Create ML training job
                ml_job = MLTrainingJob.objects.create(
                    uploaded_file=uploaded_file,
                    target_column=target_column,
                    problem_type=problem_type.upper(),
                    status='PENDING'
                )
                
                messages.success(request, f'Target column "{target_column}" selected. '
                               f'Detected as {problem_type} problem. Starting training...')
                
                # Redirect to training view
                return redirect('analytics:train_model', file_id=uploaded_file.id, target_column=target_column)
                
            except Exception as e:
                messages.error(request, f'Error analyzing target column: {str(e)}')
        
        context = {
            'uploaded_file': uploaded_file,
            'columns_info': columns_info,
            'dataset_shape': df.shape,
            'total_rows': len(df),
            'total_columns': len(df.columns)
        }
        
        return render(request, 'analytics/select_target.html', context)
        
    except Exception as e:
        messages.error(request, f'Error loading dataset: {str(e)}')
        return redirect('accounts:dashboard')


@login_required
def train_model(request, file_id, target_column):
    """
    View to handle model training
    Can be called directly or via background job (for production)
    """
    try:
        # Get or create the ML training job
        uploaded_file = get_object_or_404(UploadedFile, id=file_id, user=request.user)
        
        # Try to get existing job or create new one
        try:
            # First, try to get the most recent job for this file and target column
            ml_job = MLTrainingJob.objects.filter(
                uploaded_file=uploaded_file,
                target_column=target_column
            ).latest('created_at')
            
            # If we found an existing job, check its status
            if ml_job.status == 'COMPLETED':
                # If user explicitly wants to retrain, create a new job
                if request.GET.get('retrain') == 'true':
                    ml_job = MLTrainingJob.objects.create(
                        uploaded_file=uploaded_file,
                        target_column=target_column,
                        problem_type='PENDING',
                        status='PENDING'
                    )
                    created = True
                else:
                    # Show completed results
                    return redirect('analytics:ml_results', job_id=ml_job.id)
            elif ml_job.status == 'FAILED':
                # If previous job failed, create a new one
                ml_job = MLTrainingJob.objects.create(
                    uploaded_file=uploaded_file,
                    target_column=target_column,
                    problem_type='PENDING',
                    status='PENDING'
                )
                created = True
            else:
                # Job is pending or running, use existing
                created = False
                
        except MLTrainingJob.DoesNotExist:
            # No existing job, create a new one
            ml_job = MLTrainingJob.objects.create(
                uploaded_file=uploaded_file,
                target_column=target_column,
                problem_type='PENDING',
                status='PENDING'
            )
            created = True
        
        # If job was just created or problem_type is still PENDING, detect it
        if created or ml_job.problem_type == 'PENDING':
            try:
                df = read_file(uploaded_file.file.path)
                problem_type, target_info = detect_problem_type(df, target_column)
                ml_job.problem_type = problem_type.upper()
                ml_job.save()
            except Exception as e:
                messages.error(request, f'Error detecting problem type: {str(e)}')
                return redirect('analytics:select_target_column', file_id=uploaded_file.id)
        
        # If already completed, show results
        if ml_job.status == 'COMPLETED':
            return redirect('analytics:ml_results', job_id=ml_job.id)
        
        # If failed, show error
        if ml_job.status == 'FAILED':
            messages.error(request, f'Training failed: {ml_job.error_message}')
            return redirect('analytics:select_target_column', file_id=ml_job.uploaded_file.id)
        
        # Start training (for demo, we'll do it synchronously)
        # In production, this should be a background task
        if request.method == 'POST' or ml_job.status == 'PENDING':
            try:
                # Update status
                ml_job.status = 'RUNNING'
                ml_job.save()
                
                # Perform the training
                training_result = perform_ml_training(ml_job)
                
                if training_result['success']:
                    ml_job.status = 'COMPLETED'
                    ml_job.completed_at = timezone.now()
                    ml_job.preprocessing_info = training_result['preprocessing_info']
                    ml_job.model_results = training_result['model_results']
                    ml_job.model_paths = training_result['model_paths']
                    ml_job.plot_paths = training_result['plot_paths']
                    ml_job.best_model_name = training_result['best_model_name']
                    ml_job.best_model_score = training_result['best_model_score']
                    ml_job.save()
                    
                    messages.success(request, 'Model training completed successfully!')
                    return redirect('analytics:ml_results', job_id=ml_job.id)
                else:
                    ml_job.status = 'FAILED'
                    ml_job.error_message = training_result['error']
                    ml_job.save()
                    messages.error(request, f'Training failed: {training_result["error"]}')
                    return redirect('analytics:select_target_column', file_id=ml_job.uploaded_file.id)
                    
            except Exception as e:
                ml_job.status = 'FAILED'
                ml_job.error_message = str(e)
                ml_job.save()
                messages.error(request, f'Training failed: {str(e)}')
                return redirect('analytics:select_target_column', file_id=ml_job.uploaded_file.id)
        
        # Show training progress page
        context = {
            'ml_job': ml_job,
            'uploaded_file': ml_job.uploaded_file
        }
        
        return render(request, 'analytics/train_model.html', context)
        
    except Exception as e:
        messages.error(request, f'Error in training process: {str(e)}')
        return redirect('accounts:dashboard')


def perform_ml_training(ml_job):
    """
    Core function that performs the actual ML training
    This can be moved to a background task (Celery) for production
    """
    try:
        # Read the dataset
        df = read_file(ml_job.uploaded_file.file.path)
        
        # Detect problem type (double-check)
        problem_type, target_info = detect_problem_type(df, ml_job.target_column)
        
        # Preprocess the data
        preprocessing_result = preprocess_data(df, ml_job.target_column, problem_type)
        
        # Train models
        training_result = train_models(
            preprocessing_result['X'], 
            preprocessing_result['y'], 
            problem_type,
            test_size=ml_job.test_size,
            random_state=ml_job.random_state
        )
        
        # Generate plots
        plot_paths = generate_ml_plots(
            training_result, 
            problem_type, 
            ml_job.uploaded_file.id,
            preprocessing_result['feature_names']
        )
        
        # Save model artifacts
        model_paths = save_model_artifacts(
            training_result['models'],
            preprocessing_result,
            training_result,
            ml_job.uploaded_file.id,
            ml_job.target_column
        )
        
        # Determine best model
        if problem_type == 'regression':
            best_model_name = min(training_result['results'].keys(), 
                                key=lambda x: training_result['results'][x]['rmse'])
            best_model_score = training_result['results'][best_model_name]['r2']
        else:
            best_model_name = max(training_result['results'].keys(), 
                                key=lambda x: training_result['results'][x]['accuracy'])
            best_model_score = training_result['results'][best_model_name]['accuracy']
        
        # Prepare serializable results
        serializable_results = {}
        for model_name, result in training_result['results'].items():
            model_result = {}
            for key, value in result.items():
                if key not in ['predictions', 'model']:  # Remove non-serializable items
                    model_result[key] = convert_to_serializable(value)
            serializable_results[model_name] = model_result
        
        return {
            'success': True,
            'preprocessing_info': convert_to_serializable({
                'steps': preprocessing_result['preprocessing_steps'],
                'feature_count': preprocessing_result['final_info']['feature_count'],
                'sample_count': preprocessing_result['final_info']['sample_count'],
                'removed_columns': preprocessing_result['removed_columns'],
                'target_info': target_info
            }),
            'model_results': serializable_results,
            'model_paths': model_paths,
            'plot_paths': plot_paths,
            'best_model_name': best_model_name,
            'best_model_score': convert_to_serializable(best_model_score)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@login_required
def ml_results(request, job_id):
    """
    View to display ML training results
    Shows metrics, plots, and model comparison
    """
    try:
        # Get the completed ML job
        ml_job = get_object_or_404(MLTrainingJob, id=job_id, uploaded_file__user=request.user)
        
        if ml_job.status != 'COMPLETED':
            messages.warning(request, 'Training not completed yet.')
            return redirect('analytics:train_model', job_id=job_id)
        
        # Prepare context for template
        context = {
            'ml_job': ml_job,
            'uploaded_file': ml_job.uploaded_file,
            'problem_type': ml_job.problem_type.lower(),
            'target_column': ml_job.target_column,
            'preprocessing_info': ml_job.preprocessing_info,
            'model_results': ml_job.model_results,
            'plot_paths': ml_job.plot_paths,
            'best_model': {
                'name': ml_job.best_model_name,
                'score': ml_job.best_model_score
            },
            'training_time': (ml_job.completed_at - ml_job.created_at).total_seconds() if ml_job.completed_at else None
        }
        
        return render(request, 'analytics/ml_results.html', context)
        
    except Exception as e:
        messages.error(request, f'Error loading results: {str(e)}')
        return redirect('accounts:dashboard')


@login_required
def ml_jobs_list(request):
    """
    View to list all ML training jobs for the current user
    """
    try:
        # Get all ML jobs for user's files
        ml_jobs = MLTrainingJob.objects.filter(
            uploaded_file__user=request.user
        ).order_by('-created_at')
        
        context = {
            'ml_jobs': ml_jobs
        }
        
        return render(request, 'analytics/ml_jobs_list.html', context)
        
    except Exception as e:
        messages.error(request, f'Error loading ML jobs: {str(e)}')
        return redirect('accounts:dashboard')


@login_required
def job_status_api(request, job_id):
    """
    API endpoint to check job status (for AJAX polling)
    """
    try:
        ml_job = get_object_or_404(MLTrainingJob, id=job_id, uploaded_file__user=request.user)
        
        return JsonResponse({
            'status': ml_job.status,
            'progress': {
                'PENDING': 0,
                'RUNNING': 50,
                'COMPLETED': 100,
                'FAILED': 0
            }.get(ml_job.status, 0),
            'error_message': ml_job.error_message,
            'completed': ml_job.status == 'COMPLETED'
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'ERROR',
            'error_message': str(e)
        }, status=500)
