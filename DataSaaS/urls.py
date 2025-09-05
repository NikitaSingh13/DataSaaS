from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from .test_views import test_media_files  # ✅ Import test view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include(("accounts.urls", "accounts"), namespace="accounts")),  # delegate to accounts app with namespace
    path('analytics/', include(("analytics.urls", "analytics"), namespace="analytics")),  # analytics app URLs with namespace
    
    # ✅ Test endpoint for debugging media files
    path('test-media/', test_media_files, name='test_media'),

    #allauth
    path("accounts/", include("allauth.urls"))
]

# Serve media files in both development and production
# This is essential for EDA plots and ML visualizations to work on Render
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Also serve static files (fallback for production)
if not settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
