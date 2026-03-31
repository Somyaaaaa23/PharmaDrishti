@echo off
echo Starting PharmaDrishti Dashboard...
echo.
echo Make sure you have:
echo 1. Installed dependencies: pip install -r requirements.txt
echo 2. Configured Gemini API key (optional) in .streamlit/secrets.toml
echo.
echo Dashboard will open at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run dashboard.py
git add .