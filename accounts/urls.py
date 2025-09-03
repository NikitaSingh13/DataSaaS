from django.urls import path
from . import views

app_name = "accounts" 

urlpatterns = [
    # Authentication URLs
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    
    # Dashboard and File Management URLs
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('upload/', views.upload_file, name='upload_file'),
    path('download/<int:file_id>/', views.download_file, name='download_file'),
    path('delete/<int:file_id>/', views.delete_file, name='delete_file'),
]
