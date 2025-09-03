from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.http import HttpResponse, Http404
from django.db.models import Count
from .forms import UploadFileForm
from .models import UploadedFile, Insight, Report
import pandas as pd
import os
import mimetypes


def home(request):
    """Home page view"""
    return render(request, 'accounts/home.html')


def login_view(request):
    """Login view"""
    if request.user.is_authenticated:
        return redirect('accounts:dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        
        if not username or not password:
            messages.error(request, 'Please enter both username/email and password.')
            return render(request, 'accounts/login.html')
        
        # Try to authenticate with username first, then email
        user = authenticate(request, username=username, password=password)
        
        if user is None:
            # Try to find user by email
            try:
                user_obj = User.objects.get(email=username)
                user = authenticate(request, username=user_obj.username, password=password)
            except User.DoesNotExist:
                user = None
        
        if user is not None:
            login(request, user)
            # Redirect to next parameter if available, otherwise dashboard
            next_url = request.GET.get('next', 'accounts:dashboard')
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid username/email or password.')
    
    return render(request, 'accounts/login.html')


def signup_view(request):
    """Signup view"""
    if request.user.is_authenticated:
        return redirect('accounts:dashboard')
    
    if request.method == 'POST':
        first_name = request.POST.get('first_name', '').strip()
        last_name = request.POST.get('last_name', '').strip()
        email = request.POST.get('email', '').strip()
        username = request.POST.get('username', '').strip()
        password1 = request.POST.get('password1', '')
        password2 = request.POST.get('password2', '')
        
        # Basic validation
        if not all([first_name, last_name, email, username, password1, password2]):
            messages.error(request, 'All fields are required.')
            return render(request, 'accounts/signup.html')
        
        # Validate email format
        try:
            validate_email(email)
        except ValidationError:
            messages.error(request, 'Please enter a valid email address.')
            return render(request, 'accounts/signup.html')
        
        # Password validation
        if password1 != password2:
            messages.error(request, 'Passwords do not match.')
            return render(request, 'accounts/signup.html')
        
        if len(password1) < 8:
            messages.error(request, 'Password must be at least 8 characters long.')
            return render(request, 'accounts/signup.html')
        
        # Check if username already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists.')
            return render(request, 'accounts/signup.html')
        
        # Check if email already exists
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already exists.')
            return render(request, 'accounts/signup.html')
        
        try:
            # Create user
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password1,
                first_name=first_name,
                last_name=last_name
            )
            
            # Login the user automatically
            login(request, user)
            messages.success(request, 'Account created successfully!')
            return redirect('accounts:dashboard')
            
        except Exception as e:
            messages.error(request, f'Error creating account: {str(e)}')
    
    return render(request, 'accounts/signup.html')


@login_required
def dashboard_view(request):
    """Dashboard view for authenticated users with dynamic stats"""
    # Get user's uploaded files
    user_files = UploadedFile.objects.filter(user=request.user)[:5]  # Show latest 5 files
    
    # Calculate dynamic stats
    total_files = UploadedFile.objects.filter(user=request.user).count()
    total_insights = Insight.objects.filter(user=request.user).count()
    total_reports = Report.objects.filter(user=request.user).count()
    
    context = {
        'user_files': user_files,
        'total_files': total_files,
        'total_insights': total_insights,
        'total_reports': total_reports,
    }
    
    return render(request, 'accounts/dashboard.html', context)


@login_required
def upload_file(request):
    """Handle file upload and preview first 5 rows using Pandas"""
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        
        if form.is_valid():
            try:
                # Save the uploaded file
                uploaded_file = form.save(user=request.user)
                
                # Try to read and preview the file using pandas
                file_path = uploaded_file.file.path
                preview_data = None
                error_message = None
                
                try:
                    # Determine file type and read accordingly
                    file_ext = os.path.splitext(uploaded_file.filename)[1].lower()
                    
                    if file_ext == '.csv':
                        df = pd.read_csv(file_path)
                    elif file_ext in ['.xlsx', '.xls']:
                        df = pd.read_excel(file_path)
                    else:
                        raise ValueError("Unsupported file format")
                    
                    # Get preview data (first 5 rows) - Convert to JSON serializable format
                    preview_data = {
                        'columns': [str(col) for col in df.columns],  # Convert to strings
                        'rows': [[str(cell) if pd.notna(cell) else '' for cell in row] for row in df.head(5).values],  # Convert to strings
                        'total_rows': int(len(df)),  # Convert to Python int
                        'total_columns': int(len(df.columns))  # Convert to Python int
                    }
                    
                    # Create a basic insight record with JSON-safe data
                    Insight.objects.create(
                        user=request.user,
                        uploaded_file=uploaded_file,
                        insight_type='file_preview',
                        insight_data=preview_data
                    )
                    
                except Exception as e:
                    error_message = f"Error reading file: {str(e)}"
                
                context = {
                    'uploaded_file': uploaded_file,
                    'preview_data': preview_data,
                    'error_message': error_message,
                    'form': UploadFileForm()  # Reset form
                }
                
                messages.success(request, f'File "{uploaded_file.filename}" uploaded successfully!')
                return render(request, 'accounts/upload.html', context)
                
            except Exception as e:
                messages.error(request, f'Error uploading file: {str(e)}')
                form = UploadFileForm()
        else:
            # Form has validation errors
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
    else:
        form = UploadFileForm()
    
    context = {'form': form}
    return render(request, 'accounts/upload.html', context)


@login_required
def download_file(request, file_id):
    """Download uploaded file"""
    uploaded_file = get_object_or_404(UploadedFile, id=file_id, user=request.user)
    
    try:
        # Create a report record for tracking
        Report.objects.create(
            user=request.user,
            uploaded_file=uploaded_file,
            report_type='File Download'
        )
        
        # Serve the file
        file_path = uploaded_file.file.path
        
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fh:
                response = HttpResponse(fh.read(), content_type="application/octet-stream")
                response['Content-Disposition'] = f'attachment; filename="{uploaded_file.filename}"'
                return response
        else:
            messages.error(request, 'File not found.')
            return redirect('accounts:dashboard')
    
    except Exception as e:
        messages.error(request, f'Error downloading file: {str(e)}')
        return redirect('accounts:dashboard')


@login_required
def delete_file(request, file_id):
    """Delete uploaded file"""
    uploaded_file = get_object_or_404(UploadedFile, id=file_id, user=request.user)
    
    try:
        # Delete the physical file
        if uploaded_file.file and os.path.exists(uploaded_file.file.path):
            os.remove(uploaded_file.file.path)
        
        # Delete the database record
        filename = uploaded_file.filename
        uploaded_file.delete()
        
        messages.success(request, f'File "{filename}" deleted successfully!')
        
    except Exception as e:
        messages.error(request, f'Error deleting file: {str(e)}')
    
    return redirect('accounts:dashboard')


def logout_view(request):
    """Logout view"""
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('accounts:home')