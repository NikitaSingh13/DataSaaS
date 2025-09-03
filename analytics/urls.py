from django.urls import path
from . import views
from . import ml_views

app_name = 'analytics'

urlpatterns = [
    # EDA Analysis URLs
    path('eda/<int:file_id>/', views.eda_view, name='eda_view'),
    path('column-analysis/<int:file_id>/', views.column_analysis_view, name='column_analysis'),
    
    # ML Training URLs
    path('ml/select-target/<int:file_id>/', ml_views.select_target_column, name='select_target_column'),
    path('ml/train/<int:file_id>/<str:target_column>/', ml_views.train_model, name='train_model'),
    path('ml/results/<int:job_id>/', ml_views.ml_results, name='ml_results'),
    path('ml/jobs/', ml_views.ml_jobs_list, name='ml_jobs_list'),
]
