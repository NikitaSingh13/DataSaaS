"""
Custom middleware for serving media files in production
This is needed for platforms like Render where media files need to be served by Django
"""
import os
from django.conf import settings
from django.http import Http404, FileResponse
from django.utils.deprecation import MiddlewareMixin


class MediaFilesMiddleware(MiddlewareMixin):
    """
    Middleware to serve media files in production
    """
    
    def process_request(self, request):
        """
        Serve media files in production when DEBUG=False
        """
        if not settings.DEBUG and request.path.startswith(settings.MEDIA_URL):
            # Get the file path
            file_path = request.path[len(settings.MEDIA_URL):]
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)
            
            # Check if file exists
            if os.path.exists(full_path) and os.path.isfile(full_path):
                return FileResponse(
                    open(full_path, 'rb'),
                    as_attachment=False
                )
            else:
                raise Http404("Media file not found")
        
        return None
