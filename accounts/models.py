from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
import os


class UserProfile(models.Model):
    """User profile model to extend Django's User model with additional fields"""
    ROLE_CHOICES = (
        ('FREE', 'Free'),
        ('PREMIUM', 'Premium'),
    )
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)  # link each user to one profile
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='FREE')

    def __str__(self):
        return f"{self.user.username} - {self.role}"


class UploadedFile(models.Model):
    """Model to store uploaded CSV/Excel files for data analysis"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_files')
    file = models.FileField(upload_to='uploads/%Y/%m/%d/')
    filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_size = models.BigIntegerField(null=True, blank=True)  # File size in bytes
    file_type = models.CharField(max_length=50, null=True, blank=True)  # CSV, XLSX, etc.
    
    class Meta:
        ordering = ['-uploaded_at']  # Show newest files first
    
    def __str__(self):
        return f"{self.filename} - {self.user.username}"
    
    def get_file_size_display(self):
        """Return human-readable file size"""
        if not self.file_size:
            return "Unknown"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if self.file_size < 1024.0:
                return f"{self.file_size:.1f} {unit}"
            self.file_size /= 1024.0
        return f"{self.file_size:.1f} TB"


class Insight(models.Model):
    """Model to store generated insights from data analysis"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='insights')
    uploaded_file = models.ForeignKey(UploadedFile, on_delete=models.CASCADE, related_name='insights')
    insight_type = models.CharField(max_length=100)  # e.g., "correlation", "summary_stats", etc.
    insight_data = models.JSONField()  # Store insight results as JSON
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.insight_type} - {self.uploaded_file.filename}"


class Report(models.Model):
    """Model to track downloaded reports"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reports')
    uploaded_file = models.ForeignKey(UploadedFile, on_delete=models.CASCADE, related_name='reports')
    report_type = models.CharField(max_length=100)  # e.g., "PDF", "Excel", "Chart"
    downloaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.report_type} - {self.uploaded_file.filename}"


# 🔔 Signals to auto-create and save profile
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.userprofile.save()
