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


class MLTrainingJob(models.Model):
    """Model to track ML training jobs and their status"""
    STATUS_CHOICES = (
        ('PENDING', 'Pending'),
        ('RUNNING', 'Running'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    )
    
    PROBLEM_TYPE_CHOICES = (
        ('REGRESSION', 'Regression'),
        ('CLASSIFICATION', 'Classification'),
    )
    
    uploaded_file = models.ForeignKey('UploadedFile', on_delete=models.CASCADE)
    target_column = models.CharField(max_length=255)
    problem_type = models.CharField(max_length=20, choices=PROBLEM_TYPE_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    
    # Training configuration
    test_size = models.FloatField(default=0.2)
    random_state = models.IntegerField(default=42)
    
    # Results (JSON fields)
    preprocessing_info = models.JSONField(null=True, blank=True)
    model_results = models.JSONField(null=True, blank=True)
    model_paths = models.JSONField(null=True, blank=True)  # Paths to saved models
    plot_paths = models.JSONField(null=True, blank=True)   # Paths to generated plots
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    
    # Performance metrics
    best_model_name = models.CharField(max_length=100, null=True, blank=True)
    best_model_score = models.FloatField(null=True, blank=True)  # R² for regression, Accuracy for classification
    
    def __str__(self):
        return f"ML Job {self.id} - {self.uploaded_file.filename} - {self.target_column}"
    
    class Meta:
        ordering = ['-created_at']
        # Prevent duplicate jobs for the same file and target column
        # Note: This allows multiple training attempts but requires explicit deletion of old jobs


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
