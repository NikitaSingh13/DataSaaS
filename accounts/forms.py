from django import forms
from .models import UploadedFile
import os


class UploadFileForm(forms.ModelForm):
    """Form for uploading CSV and Excel files"""
    
    class Meta:
        model = UploadedFile
        fields = ['file']
        widgets = {
            'file': forms.FileInput(attrs={
                'class': 'file-input file-input-bordered file-input-primary w-full',
                'accept': '.csv,.xlsx,.xls',
                'id': 'file-upload'
            })
        }
    
    def clean_file(self):
        """Validate uploaded file"""
        file = self.cleaned_data.get('file')
        
        if not file:
            raise forms.ValidationError("Please select a file to upload.")
        
        # Check file extension
        ext = os.path.splitext(file.name)[1].lower()
        if ext not in ['.csv', '.xlsx', '.xls']:
            raise forms.ValidationError(
                "Only CSV and Excel files (.csv, .xlsx, .xls) are allowed."
            )
        
        # Check file size (max 100MB)
        max_size = 100 * 1024 * 1024  # 100MB in bytes
        if file.size > max_size:
            raise forms.ValidationError(
                f"File size too large. Maximum allowed size is 100MB. "
                f"Your file is {file.size / (1024*1024):.1f}MB."
            )
        
        return file
    
    def save(self, commit=True, user=None):
        """Save the form with additional fields"""
        instance = super().save(commit=False)
        
        if user:
            instance.user = user
        
        # Set filename and file metadata
        if instance.file:
            instance.filename = instance.file.name
            instance.file_size = instance.file.size
            instance.file_type = os.path.splitext(instance.file.name)[1].upper()[1:]  # Remove dot, uppercase
        
        if commit:
            instance.save()
        
        return instance
