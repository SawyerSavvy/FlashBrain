# Project Decomposition Agent - Optimized for Cloud Run
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Run the application
CMD exec uvicorn __main__:app --host 0.0.0.0 --port ${PORT} --workers 1
