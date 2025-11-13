#!/bin/bash
# BMWithAE Backend Startup Script

echo "Starting BMWithAE Backend..."

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Warning: Not running in a virtual environment"
    echo "Please activate your environment first:"
    echo "  conda activate bmwithae"
    echo "  OR"
    echo "  source venv/bin/activate"
    exit 1
fi

# Start Flask server
python app.py



