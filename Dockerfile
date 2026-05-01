# Base image for Flask app
# Use CUDA image if GPU inference fallback is desired; else python:3.10-slim
FROM python:3.10-slim

# Metadata
LABEL maintainer="CultiKure Team"
LABEL description="CultiKure Plant Disease Detection — Flask Application"

WORKDIR /app

# System dependencies for Pillow, OpenCV-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies first (layer-caching optimisation)
COPY App/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY App/ .

# Create upload directory
RUN mkdir -p static/uploads

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/')" || exit 1

# Production server with gunicorn
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "4", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--log-level", "info", \
     "app:app"]
