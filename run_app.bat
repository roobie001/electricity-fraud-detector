@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 goto error
)
echo Installing or checking required packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error
echo Starting Electricity Anomaly Detector...
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto error
goto end
:error
echo.
echo Could not start. Install Python 3.11-3.13 and ensure internet is available for first-time package setup.
pause
:end
endlocal
