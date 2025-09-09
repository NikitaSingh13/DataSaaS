#  DataSaaS- Automated EDA & ML Platform

![DataSaaS](https://img.shields.io/badge/Django-5.2.5-green) ![Python](https://img.shields.io/badge/Python-3.13-blue) ![TailwindCSS](https://img.shields.io/badge/TailwindCSS-Latest-cyan)

A powerful Django-based SaaS platform for data analytics and machine learning. Upload your datasets and get instant insights through exploratory data analysis (EDA) and automated machine learning model training.

## 🚀 Features

### 📊 Data Analytics
- **File Upload System**: Support for CSV and Excel files
- **Exploratory Data Analysis (EDA)**: Comprehensive statistical analysis
- **Data Visualizations**: Automatic generation of plots and charts
- **Column Analysis**: Detailed insights into individual columns
- **Missing Data Analysis**: Identify and visualize missing values

### 🤖 Machine Learning
- **Automatic Problem Detection**: Classifies regression vs classification problems
- **Smart Preprocessing**: Handles missing values, categorical encoding, feature selection
- **Multiple Model Training**: Baseline + 2 advanced models (Linear/Logistic Regression + Random Forest)
- **Model Comparison**: Side-by-side performance metrics
- **Visualization**: Training plots, confusion matrices, feature importance
- **Memory Optimization**: Intelligent handling of high-cardinality data

### 🎨 User Interface
- **Modern Design**: Beautiful gradient-based UI with TailwindCSS
- **Responsive Layout**: Works on desktop and mobile devices
- **Interactive Dashboard**: Real-time file management and analytics
- **Progress Tracking**: Visual feedback for ML training progress

### 🔐 Authentication
- **Custom User System**: Secure login and registration
- **User Profiles**: Personal dashboards and file management
- **Session Management**: Secure user sessions

## 🛠️ Tech Stack

- **Backend**: Django 5.2.5
- **Database**: SQLite (development) / PostgreSQL (production ready)
- **Data Science**: pandas, numpy, matplotlib, seaborn, scikit-learn
- **Frontend**: TailwindCSS, DaisyUI, Font Awesome
- **File Processing**: openpyxl, xlrd for Excel support
- **ML Visualization**: plotly for interactive charts

## 📦 Installation

### Prerequisites
- Python 3.13+
- Node.js (for TailwindCSS)
- Git

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd DataSaaS
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Mac/Linux
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   SECRET_KEY=your-django-secret-key
   DEBUG=True
   DATABASE_URL=sqlite:///db.sqlite3
   ```

5. **Run database migrations**
   ```bash
   python manage.py migrate
   ```

6. **Create superuser (optional)**
   ```bash
   python manage.py createsuperuser
   ```

7. **Install TailwindCSS dependencies**
   ```bash
   python manage.py tailwind install
   ```

8. **Start the development server**
   ```bash
   # Terminal 1: Django server
   python manage.py runserver
   
   # Terminal 2: TailwindCSS watcher (in development)
   python manage.py tailwind start
   ```

9. **Open your browser**
   Navigate to `http://127.0.0.1:8000`

## 📱 Usage

### Getting Started
1. **Sign Up**: Create a new account or login
2. **Upload Data**: Upload your CSV or Excel files
3. **Analyze**: Click "Analyze" to perform EDA
4. **Train Models**: Click "ML" to train machine learning models

### EDA Features
- **Dataset Overview**: Basic statistics and data types
- **Missing Values**: Visualize missing data patterns
- **Distributions**: Histograms and box plots
- **Correlations**: Correlation heatmaps
- **Categorical Analysis**: Value counts and distributions

### ML Training Process
1. **Target Selection**: Choose your target column
2. **Auto Preprocessing**: System automatically cleans and prepares data
3. **Model Training**: Trains baseline + 2 advanced models
4. **Results**: Compare model performance with metrics and plots

## 🏗️ Project Structure

```
DataSaaS/
├── accounts/                 # User authentication and file management
│   ├── models.py            # User, File, ML Job models
│   ├── views.py             # Authentication and dashboard views
│   └── templates/           # Login, signup, dashboard templates
├── analytics/               # Data analytics and ML functionality
│   ├── utils/
│   │   ├── eda_utils.py     # EDA analysis functions
│   │   └── ml_utils.py      # ML training pipeline
│   ├── views.py             # EDA views
│   ├── ml_views.py          # ML training views
│   └── templatetags/        # Custom template filters
├── templates/               # Base templates and layouts
├── static/                  # Static files (CSS, JS, images)
├── media/                   # User uploads and generated plots
├── theme/                   # TailwindCSS theme configuration
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore patterns
└── manage.py               # Django management script
```

## 🔧 Configuration

### Environment Variables
```env
SECRET_KEY=your-secret-key
DEBUG=True/False
DATABASE_URL=your-database-url
```

### Database Setup
**Development**: SQLite (default)
**Production**: PostgreSQL (recommended)

```python
# For PostgreSQL
DATABASE_URL=postgresql://user:password@host:port/database
```

## 📊 Supported Data Formats

- **CSV files** (.csv)
- **Excel files** (.xlsx, .xls)
- **Automatic encoding detection**
- **Memory-efficient processing**

## 🤖 ML Capabilities

### Problem Types
- **Regression**: Numerical target prediction
- **Classification**: Category prediction

### Models Included
1. **Baseline Models**: Simple statistical baselines
2. **Linear Models**: Linear/Logistic Regression
3. **Ensemble Models**: Random Forest

### Preprocessing Features
- Missing value imputation
- Categorical encoding (with high-cardinality handling)
- Feature selection
- Data type optimization
- Memory usage monitoring

## 🎨 UI Features

- **Modern Gradients**: Beautiful color schemes
- **Responsive Design**: Mobile-friendly interface
- **Interactive Elements**: Hover effects and animations
- **Progress Indicators**: Real-time feedback
- **Data Tables**: Sortable and searchable tables

## 🚀 Deployment

### Production Checklist
1. Set `DEBUG=False` in production
2. Use PostgreSQL database
3. Set up proper static file serving
4. Configure ALLOWED_HOSTS
5. Use environment variables for secrets
6. Set up HTTPS

### Docker Deployment (Optional)
```dockerfile
# Dockerfile example
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues:
1. Check the [Issues](../../issues) page
2. Create a new issue with detailed description
3. Include error messages and steps to reproduce

## 🙏 Acknowledgments

- Django framework for the robust backend
- TailwindCSS for the beautiful UI components
- scikit-learn for machine learning capabilities
- The open-source community for amazing libraries

---

**Built with ❤️ using Django and modern web technologies**
