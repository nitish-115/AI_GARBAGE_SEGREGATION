@echo off
:: ============================================================
::  run.bat  —  Quick launcher (Windows)
:: ============================================================

echo.
echo   ♻️  AI Garbage Segregation System
echo   ─────────────────────────────────
echo.

:: Create venv if missing
if not exist venv (
    echo [setup] Creating virtual environment ...
    python -m venv venv
)

:: Activate
call venv\Scripts\activate

:: Install deps
echo [setup] Installing dependencies ...
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo.
echo   Choose an action:
echo   1) Train model
echo   2) Launch web app (Streamlit)
echo   3) Live webcam detection
echo   4) Predict single image
echo.
set /p CHOICE="  Enter choice [1-4]: "

if "%CHOICE%"=="1" (
    echo [train] Starting training ...
    python train.py
) else if "%CHOICE%"=="2" (
    echo [app] Launching Streamlit ...
    streamlit run app.py
) else if "%CHOICE%"=="3" (
    echo [webcam] Starting webcam ...
    python webcam_detection.py
) else if "%CHOICE%"=="4" (
    set /p IMG_PATH="  Image path: "
    python predict.py %IMG_PATH%
) else (
    echo Invalid choice.
)

pause
