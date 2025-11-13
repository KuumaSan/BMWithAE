#!/bin/bash
# Production deployment script for BMWithAE backend
# For Linux/Mac systems

# Set environment variables
export FLASK_ENV=production
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5001

# Number of worker processes (recommended: 2-4 x CPU cores)
WORKERS=4

# Timeout for worker processes (seconds)
TIMEOUT=120

# Access log file
ACCESS_LOG="logs/access.log"
ERROR_LOG="logs/error.log"

# Create log directory if it doesn't exist
mkdir -p logs

echo "========================================"
echo "Starting BMWithAE Backend (Production)"
echo "========================================"
echo "Workers: $WORKERS"
echo "Timeout: ${TIMEOUT}s"
echo "Host: $FLASK_HOST:$FLASK_PORT"
echo "========================================"

# Start Gunicorn
gunicorn \
    --workers $WORKERS \
    --worker-class sync \
    --timeout $TIMEOUT \
    --bind $FLASK_HOST:$FLASK_PORT \
    --access-logfile $ACCESS_LOG \
    --error-logfile $ERROR_LOG \
    --log-level info \
    --preload \
    wsgi:app

# Alternative: Using Waitress (cross-platform)
# waitress-serve --host=$FLASK_HOST --port=$FLASK_PORT --threads=8 wsgi:app

