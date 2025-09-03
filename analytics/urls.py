from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    # EDA Analysis URLs
    path('eda/<int:file_id>/', views.eda_view, name='eda_view'),
    path('column-analysis/<int:file_id>/', views.column_analysis_view, name='column_analysis'),
]
