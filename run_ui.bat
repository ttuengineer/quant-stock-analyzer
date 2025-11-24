@echo off
echo Starting Stock Analyzer Dashboard...
echo.
echo Opening browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

cd /d %~dp0
call .venv\Scripts\activate
streamlit run app.py
