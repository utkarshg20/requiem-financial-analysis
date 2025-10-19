#!/bin/bash

# Start Requiem UI and API servers
echo "🚀 Starting Requiem Platform..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if API is already running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Requiem API already running on port 8000"
else
    echo "🔄 Starting Requiem API on port 8000..."
    uvicorn api.main:app --reload --port 8000 &
    API_PID=$!
    sleep 3
fi

# Check if UI server is already running
if curl -s http://localhost:3001 > /dev/null 2>&1; then
    echo "✅ UI server already running on port 3001"
else
    echo "🔄 Starting UI server on port 3001..."
    cd ui
    python3 server.py 3001 &
    UI_PID=$!
    cd ..
fi

echo ""
echo "🎉 Requiem Platform is ready!"
echo ""
echo "📊 UI: http://localhost:3001"
echo "🔗 API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for interrupt
trap 'echo ""; echo "🛑 Stopping servers..."; kill $API_PID $UI_PID 2>/dev/null; exit 0' INT

# Keep script running
wait
