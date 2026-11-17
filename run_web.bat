@echo off
echo Menjalankan Palm Detection Web Console (Headless)...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" web.py --host 0.0.0.0 --port 5000
) else if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" web.py --host 0.0.0.0 --port 5000
) else (
    python web.py --host 0.0.0.0 --port 5000
)
