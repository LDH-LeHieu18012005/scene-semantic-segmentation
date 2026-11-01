@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
set "DEMO_PYTHON=.venv\Scripts\python.exe"
if exist ".python-runtime\python.exe" set "DEMO_PYTHON=.python-runtime\python.exe"
if not exist "%DEMO_PYTHON%" (
  echo Chua co moi truong Python. Xem demo\README.md de cai dat.
  pause
  exit /b 1
)
if not exist "checkpoints\b7_seed17_last.pt" (
  echo Chua co checkpoint. Hay chay DOWNLOAD_MODEL.cmd truoc.
  pause
  exit /b 1
)
echo.
echo PSP Studio - demo CPU
echo Mo trinh duyet tai: http://127.0.0.1:8765
echo Giu cua so nay mo khi demo. Nhan Ctrl+C de dung.
echo.
"%DEMO_PYTHON%" demo\server.py
if errorlevel 1 pause
