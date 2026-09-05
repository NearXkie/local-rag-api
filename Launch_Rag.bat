

@echo off
:: Set the title of the launcher window
title Local RAG Launcher
echo ===================================================
echo   🚀 STARTING YOUR LOCAL OFFLINE RAG SYSTEM 🚀
echo ===================================================
echo.

:: 1. Navigate to your D: drive project folder
cd /d "D:\Tools\Projects\local-rag-api"

:: 2. Launch FastAPI in a separate minimized or background window
echo 🔌 Launching FastAPI Backend on Port 8000...
start "RAG API Backend" /min cmd /c "call .venv\Scripts\activate && python -m uvicorn src.api.main:app"

:: 3. Give the backend 2 seconds to establish the ChromaDB connection
timeout /t 2 /nobreak >nul

:: 4. Launch Streamlit in a separate window (this will open your browser automatically)
echo 🎨 Launching Streamlit Frontend Dashboard...
start "RAG Streamlit Frontend" cmd /c "call .venv\Scripts\activate && python -m streamlit run src/ui/app.py"

echo.
echo ===================================================
echo   🎉 Success! Both servers are spinning up.
echo   You can minimize these terminal windows now.
echo ===================================================
timeout /t 3 >nul
exit