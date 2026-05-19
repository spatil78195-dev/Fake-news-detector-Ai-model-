@echo off
title Fake News Detector - One-Click Startup
color 0A

echo.
echo  =======================================================
echo    Fake News Detector - Auto Setup ^& Launch
echo  =======================================================
echo.

REM ── Change to the project directory ────────────────────────────────────────
cd /d "%~dp0"

REM ── Check that venv exists ─────────────────────────────────────────────────
if not exist "venv\Scripts\python.exe" (
    echo  [ERROR] Virtual environment not found!
    echo  Please run:  python -m venv venv
    echo  Then:        venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo  [1/5] Installing missing packages...
venv\Scripts\pip install --quiet flask pandas numpy scikit-learn nltk python-dotenv
venv\Scripts\pip install --quiet apscheduler pymongo newsapi-python
venv\Scripts\pip install --quiet Pillow requests beautifulsoup4 lxml langdetect deep-translator SpeechRecognition
venv\Scripts\pip install --quiet newspaper3k 2>nul
venv\Scripts\pip install --quiet imagehash 2>nul
venv\Scripts\pip install --quiet opencv-python-headless 2>nul
venv\Scripts\pip install --quiet pydub 2>nul
echo  [1/5] Done.
echo.

echo  [2/5] Downloading NLTK data...
venv\Scripts\python.exe -c "import nltk; nltk.download('stopwords',quiet=True); nltk.download('punkt',quiet=True); print('  NLTK ready.')"
echo.

echo  [3/5] Generating / verifying dataset...
venv\Scripts\python.exe generate_dataset.py
echo.

echo  [4/5] Training model (may take 1-2 minutes)...
venv\Scripts\python.exe train_model.py
echo.

echo  =======================================================
echo   Server starting at:
echo     Main App:        http://localhost:5000
echo     Admin Dashboard: http://localhost:5000/admin
echo     Health Check:    http://localhost:5000/health
echo  =======================================================
echo.
echo  Press Ctrl+C to stop the server.
echo.

REM ── Start Flask server ─────────────────────────────────────────────────────
start "" "http://localhost:5000"
venv\Scripts\python.exe app.py

pause
