@echo off
REM PhonemeIQ Development Setup Script for Windows
REM Harvard-LUMS Speech Processing Laboratory

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  PhonemeIQ v3.0 - Development Environment Setup            ║
echo ║  Speech Processing Laboratory                              ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Check Python version
echo ✓ Checking Python version...
python --version

REM Create virtual environment
echo.
echo ✓ Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo   Virtual environment created
) else (
    echo   Virtual environment already exists
)

REM Activate virtual environment
echo.
echo ✓ Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo ✓ Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo ✓ Installing dependencies...
pip install -r requirements.txt

REM Verify model files
echo.
echo ✓ Checking model files...
if exist "cnn_model.h5" (
    echo   ✓ cnn_model.h5 found
) else (
    echo   ✗ WARNING: cnn_model.h5 not found
)
if exist "svm_model.pkl" (
    echo   ✓ svm_model.pkl found
) else (
    echo   ✗ WARNING: svm_model.pkl not found
)
if exist "scaler.pkl" (
    echo   ✓ scaler.pkl found
) else (
    echo   ✗ WARNING: scaler.pkl not found
)
if exist "label_encoder.pkl" (
    echo   ✓ label_encoder.pkl found
) else (
    echo   ✗ WARNING: label_encoder.pkl not found
)

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  Setup Complete!                                             ║
echo ║                                                              ║
echo ║  To start the application, run:                             ║
echo ║  $ streamlit run app_professional.py                        ║
echo ║                                                              ║
echo ║  Or use Docker:                                             ║
echo ║  $ docker-compose up                                        ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
pause
