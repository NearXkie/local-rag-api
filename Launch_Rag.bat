@echo off
title Local RAG Launcher
echo ===================================================
echo   🚀 STARTING YOUR LOCAL OFFLINE RAG SYSTEM 🚀
echo ===================================================
echo.

:: 1. Navigate to your D: drive project folder
cd /d "D:\Tools\Projects\local-rag-api"

:: 2. Launch FastAPI in a minimized command window
echo 🔌 Launching FastAPI Backend on Port 8000...
start "RAG API Backend" /min cmd /c "call .venv\Scripts\activate && python -m uvicorn src.api.main:app"

:: 3. Smart Polling: Wait for FastAPI to say "healthy" before starting Streamlit
echo ⏳ Waiting for database initialization to finish...
echo    (This will take a moment if building the database for the first time...)
echo.
set /p ="Processing database: " <nul

:wait_loop
:: Ping the FastAPI health endpoint (curl is built into modern Windows 10/11)
curl -s --max-time 1 http://127.0.0.1:8000/health 2>nul | findstr "healthy" >nul
if %errorlevel% neq 0 (
    :: Print a progress dot without starting a new line
    set /p ="." <nul
    timeout /t 3 /nobreak >nul
    goto wait_loop
)

echo.
echo.
echo ✅ Database is ready and vector store is loaded!
echo.

:: 4. Launch Streamlit (only now will your browser open!)
echo 🎨 Launching Streamlit Frontend Dashboard...
start "RAG Streamlit Frontend" cmd /c "call .venv\Scripts\activate && python -m streamlit run src/ui/app.py"

echo.
echo ===================================================
echo   🎉 Success! System is online.
echo   You can minimize this launcher now.
echo ===================================================
timeout /t 3 >nul
exit