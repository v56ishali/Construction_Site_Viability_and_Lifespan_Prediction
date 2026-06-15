# Use official slim python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Install system dependencies (needed for compiling some python packages like reportlab or scikit-learn depending on architecture)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Ensure data and logs directories exist
RUN mkdir -p data logs instance

# Run gunicorn server
# Render provides the PORT environment variable (usually 10000)
# We use 1 worker to stay within Free Tier RAM limits (512MB)
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 180 app:app
