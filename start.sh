#!/bin/bash
# Railway start script
python3 -m uvicorn api.main:app --host 0.0.0.0 --port $PORT
