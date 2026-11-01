@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
set "DEMO_PYTHON=.venv\Scripts\python.exe"
if exist ".python-runtime\python.exe" set "DEMO_PYTHON=.python-runtime\python.exe"
if not exist "%DEMO_PYTHON%" (
  echo Chua co moi truong Python. Xem demo\README.md.
  pause
  exit /b 1
)
if exist "checkpoints\b7_seed17_last.pt" (
  echo Checkpoint da ton tai. Khong ghi de.
  pause
  exit /b 0
)
if not exist "checkpoints" mkdir checkpoints
"%DEMO_PYTHON%" -m modal --profile duonglehieuhy volume get -e main pspnet-study-ade20k /results/ade20k_seed17_b16_10k_v4/b7_seed17/last.pt checkpoints/b7_seed17_last.pt.part
if errorlevel 1 (
  echo Tai khong thanh cong. Neu chua dang nhap, chay:
  echo "%DEMO_PYTHON%" -m modal setup
  pause
  exit /b 1
)
"%DEMO_PYTHON%" demo\verify_checkpoint.py checkpoints\b7_seed17_last.pt.part
if errorlevel 1 (
  echo Checkpoint khong vuot qua kiem tra. Khong doi thanh file chinh.
  pause
  exit /b 1
)
move "checkpoints\b7_seed17_last.pt.part" "checkpoints\b7_seed17_last.pt" >nul
echo Da tai va xac minh checkpoint.
pause
