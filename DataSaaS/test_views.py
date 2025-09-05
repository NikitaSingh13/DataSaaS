"""
Media Test View - For debugging media files in production
"""
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import os

def test_media_files(request):
    """
    Test view to check if media files are accessible
    """
    media_info = {
        'MEDIA_URL': settings.MEDIA_URL,
        'MEDIA_ROOT': str(settings.MEDIA_ROOT),
        'DEBUG': settings.DEBUG,
        'media_exists': os.path.exists(settings.MEDIA_ROOT),
        'plots_dir_exists': os.path.exists(os.path.join(settings.MEDIA_ROOT, 'plots')),
        'ml_plots_dir_exists': os.path.exists(os.path.join(settings.MEDIA_ROOT, 'ml_plots')),
    }
    
    # Check for any existing plot files
    if os.path.exists(settings.MEDIA_ROOT):
        try:
            plots_dir = os.path.join(settings.MEDIA_ROOT, 'plots')
            if os.path.exists(plots_dir):
                plot_files = []
                for root, dirs, files in os.walk(plots_dir):
                    for file in files:
                        if file.endswith(('.png', '.jpg', '.jpeg')):
                            rel_path = os.path.relpath(os.path.join(root, file), settings.MEDIA_ROOT)
                            plot_files.append(rel_path)
                media_info['sample_plots'] = plot_files[:5]  # First 5 files
        except Exception as e:
            media_info['error'] = str(e)
    
    return JsonResponse(media_info, json_dumps_params={'indent': 2})
