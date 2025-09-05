# DataSaaS - Data Analytics & ML Platform

![Django](https://img.shields.io/badge/Django-5.2.5-green) ![Python](https://img.shields.io/badge/Python-3.13-blue) ![TailwindCSS](https://img.shields.io/badge/TailwindCSS-Latest-cyan)

DataSaaS is a Django-based platform for data analytics and machine learning.  
Upload datasets and get instant insights through exploratory data analysis (EDA) and automated machine learning.

---

## Features

### Data Analytics
- Upload CSV and Excel files  
- Perform EDA with descriptive statistics  
- Generate data visualizations (plots, charts, heatmaps)  
- Column-level and missing value analysis  

### Machine Learning
- Detect problem type (classification/regression)  
- Automatic preprocessing (missing values, encoding, feature selection)  
- Train baseline and advanced models (Regression, Logistic Regression, Random Forest)  
- Compare models with metrics and visualizations  

### User Interface
- TailwindCSS-based modern UI  
- Responsive design for desktop and mobile  
- Interactive dashboard for managing files and results  

### Authentication
- Secure login and registration  
- User profiles with personal dashboards  

---

## Tech Stack
- **Backend**: Django 5.2.5  
- **Database**: SQLite (development), PostgreSQL (production)  
- **Data Science**: pandas, numpy, matplotlib, seaborn, scikit-learn  
- **Frontend**: TailwindCSS, DaisyUI  
- **File Handling**: openpyxl, xlrd  
- **Visualization**: plotly  

---

## Installation

### Prerequisites
- Python 3.13+  
- Node.js  
- Git  

### Setup
```bash
# Clone repo
git clone <your-repository-url>
cd DataSaaS

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Create .env file
SECRET_KEY=your-secret-key
DEBUG=True
DATABASE_URL=sqlite:///db.sqlite3

# Apply migrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser

# Install Tailwind
python manage.py tailwind install

# Run app
python manage.py runserver
python manage.py tailwind start   # in separate terminal for development


# project struucture
DataSaaS/
├── accounts/       # Authentication and dashboard
├── analytics/      # EDA and ML pipeline
├── templates/      # HTML templates
├── static/         # Static files
├── media/          # Uploaded files
├── theme/          # Tailwind config
├── requirements.txt
└── manage.py


