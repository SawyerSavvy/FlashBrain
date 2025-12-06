# RAG Service MCP Server - Optimized for Cloud Run
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# gcc: Required for building scikit-learn and other ML packages
# postgresql-client: For database connectivity
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
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

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT:-8080}/health', timeout=5)" || exit 1

# Run the application
CMD python __main__.py --host 0.0.0.0 --port ${PORT:-8080}

