#!/bin/bash
echo "Menjalankan Palm Detection Web Console (Headless) pada Nvidia Jetson / Linux..."
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python web.py --host 0.0.0.0 --port 5000
elif [ -f "../.venv/bin/python" ]; then
    ../.venv/bin/python web.py --host 0.0.0.0 --port 5000
else
    python3 web.py --host 0.0.0.0 --port 5000
fi
