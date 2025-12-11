# Use Python 3.10 which is compatible with MediaPipe
FROM python:3.10-slim

# Install system dependencies for MediaPipe and OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY climb_analyzer/ ./climb_analyzer/

# Create necessary directories
RUN mkdir -p climb_analyzer/static/uploads climb_analyzer/static/results

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=climb_analyzer/app.py
ENV PYTHONUNBUFFERED=1

# Set working directory to climb_analyzer for the app
WORKDIR /app/climb_analyzer

# Run the application
CMD ["python", "app.py"]

