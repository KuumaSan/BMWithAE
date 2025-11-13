@echo off
REM Production deployment script for BMWithAE backend
REM For Windows systems

REM Set environment variables
set FLASK_ENV=production
set FLASK_HOST=0.0.0.0
set FLASK_PORT=5001

REM Create log directory if it doesn't exist
if not exist logs mkdir logs

echo ========================================
echo Starting BMWithAE Backend (Production)
echo ========================================
echo Host: %FLASK_HOST%:%FLASK_PORT%
echo ========================================

REM Start Waitress (cross-platform WSGI server)
waitress-serve --host=%FLASK_HOST% --port=%FLASK_PORT% --threads=8 wsgi:app

