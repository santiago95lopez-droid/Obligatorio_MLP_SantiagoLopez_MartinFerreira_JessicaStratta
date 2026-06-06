@echo off
REM Start the FastAPI server in a new window and open Swagger UI.
cd /d "%~dp0"

echo Installing dependencies with Poetry if needed...
call poetry install

echo Starting API server...
start "API Server" cmd /k "poetry run uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8080"

echo Waiting for the server to start...
timeout /t 3 /nobreak > nul

start "" "http://127.0.0.1:8080/docs"
