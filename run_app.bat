@echo off
title Fake News Detector - Startup
color 0A

echo =======================================================
echo    Fake News Detector - Full Startup Script
echo =======================================================
echo.

REM ── Step 1: Upgrade pip silently ──────────────────────────────────────────
echo [1/5] Upgrading pip...
call venv\Scripts\python.exe -m pip install --upgrade pip --quiet
echo       Done.
echo.

REM ── Step 2: Install ALL required packages ─────────────────────────────────
echo [2/5] Installing required packages (this may take a few minutes)...
call venv\Scripts\pip install --quiet -r requirements-full.txt
echo       Packages installed (requirements-full.txt).
echo.

REM ── Step 3: Download NLTK data ────────────────────────────────────────────
echo [3/5] Downloading NLTK data...
call venv\Scripts\python.exe -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); print('  NLTK data ready.')"
echo.

REM ── Step 4: Generate dataset if needed and train model ────────────────────
echo [4/5] Checking dataset and model...
if not exist "dataset\Fake.csv" (
    echo       Dataset not found. Generating synthetic dataset...
    call venv\Scripts\python.exe generate_dataset.py
) else (
    echo       Dataset found.
)

if not exist "model.pkl" (
    echo       Model not found. Training model (this may take 1-2 minutes)...
    call venv\Scripts\python.exe train_model.py
) else (
    echo       Model already exists. Skipping training.
    echo       TIP: Run 'venv\Scripts\python.exe train_model.py' to retrain.
)
echo.

REM ── Step 5: Start the server ──────────────────────────────────────────────
echo [5/5] Starting the Web Application...
echo.
echo  ==> Main App:       http://localhost:5000
echo  ==> Admin Dashboard: http://localhost:5000/admin
echo  ==> Health Check:    http://localhost:5000/health
echo.
echo  Press Ctrl+C to stop the server.
echo.

call venv\Scripts\python.exe app.py

pause
