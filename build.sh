#!/usr/bin/env bash
# build.sh for Render deployment

set -o errexit  # exit on error

# Install dependencies
pip install -r requirements.txt

# Create media directories
mkdir -p media/plots
mkdir -p media/ml_plots
mkdir -p media/models
mkdir -p media/uploads

# Run Django migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --no-input

echo "Build completed successfully!"
