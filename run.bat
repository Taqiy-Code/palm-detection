@echo off
echo Menjalankan Aplikasi Deteksi Komponen Bulb Kendaraan...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py
) else (
    echo [ERROR] Virtual environment .venv tidak ditemukan.
    echo Harap jalankan uv venv atau python -m venv .venv terlebih dahulu.
    pause
)
