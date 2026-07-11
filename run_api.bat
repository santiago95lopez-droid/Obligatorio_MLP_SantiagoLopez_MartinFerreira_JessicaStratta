@echo off
setlocal EnableExtensions
REM Start the FastAPI server in a new window and open Swagger UI.
cd /d "%~dp0"

echo Verificando que Poetry esté disponible...
where poetry >nul 2>nul
if errorlevel 1 (
    echo ERROR: Poetry no se encuentra en PATH. Instale Poetry o habilite el comando en su terminal.
    exit /b 1
)

echo Configurando Poetry para crear el virtualenv dentro del proyecto...
poetry config virtualenvs.in-project true
if errorlevel 1 (
    echo ERROR: No se pudo configurar Poetry.
    exit /b 1
)

echo Instalando dependencias con Poetry...
poetry install
if errorlevel 1 (
    echo ERROR: poetry install falló.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: no se encontró el ejecutable Python en .venv\Scripts\python.exe
    exit /b 1
)

set "KERAS_MODEL_PATH=%~dp0modelohongos\modelo_hongos_mobilenet.keras"
set "TFLITE_MODEL_PATH=%~dp0modelohongos\modelo_quantizado.tflite"

if not exist "%KERAS_MODEL_PATH%" (
    echo ERROR: no se encontró el modelo Keras en %KERAS_MODEL_PATH%
    exit /b 1
)

if not exist "%TFLITE_MODEL_PATH%" (
    echo ERROR: no se encontró el modelo TFLite en %TFLITE_MODEL_PATH%
    exit /b 1
)

echo Iniciando API desde el Python del entorno local...
start "API Server" cmd /k ".venv\Scripts\python.exe -m uvicorn src.api.app:app --host 0.0.0.0 --port 8080"

echo Iniciando Streamlit en paralelo...
start "Streamlit UI" cmd /k ".venv\Scripts\python.exe -m streamlit run app_streamlit.py"

echo Waiting for the server to start...
timeout /t 3 /nobreak > nul

start "" "http://127.0.0.1:8080/docs"
