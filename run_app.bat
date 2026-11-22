@echo off
REM Intelligent Document Classification System - Windows Launcher
REM This script sets up the environment and launches the Streamlit application

echo ============================================
echo Intelligent Document Classification System
echo ============================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Creating...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo Please ensure Python is installed and in PATH.
        pause
        exit /b 1
    )

    echo Activating virtual environment...
    call venv\Scripts\activate.bat

    echo Installing dependencies...
    pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )

    echo Downloading NLTK stopwords...
    python -c "import nltk; nltk.download('stopwords', quiet=True)"
) else (
    echo Activating existing virtual environment...
    call venv\Scripts\activate.bat
)

echo.
echo Checking for trained model...
if not exist "models\document_classifier.keras" (
    echo WARNING: Trained model not found!
    echo Please run training first: python src\train.py
    echo.
    echo Do you want to run training now? (y/n)
    set /p choice=
    if /i "%choice%"=="y" (
        echo Starting training...
        python src\train.py
        if errorlevel 1 (
            echo ERROR: Training failed.
            pause
            exit /b 1
        )
        echo.
        echo Running evaluation...
        python src\evaluate.py
    ) else (
        echo Exiting. Please run training before launching the app.
        pause
        exit /b 1
    )
) else (
    echo Model found. Ready to launch.
)

echo.
echo ============================================
echo Launching Streamlit Application...
echo ============================================
echo.
echo The application will open in your default browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo.

streamlit run app\Home.py

echo.
echo Application stopped.
pause