#!/bin/bash

# Kill ports if already running (simple cleanup)
fuser -k 8000/tcp
fuser -k 5173/tcp

# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtualenv if you have one, or just run
# source .venv/bin/activate

# Start Backend
echo "Starting Backend..."
cd "$DIR"
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!

# Start Frontend
echo "Starting Frontend..."
cd "$DIR/frontend"
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for servers to start
sleep 5

# Open Browser in App Mode
# Try google-chrome, then chromium, then firefox, then xdg-open
URL="http://localhost:5173"

if command -v google-chrome &> /dev/null; then
    google-chrome --app="$URL"
elif command -v chromium-browser &> /dev/null; then
    chromium-browser --app="$URL"
else
    xdg-open "$URL"
fi

# Trap cleanup
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
}
trap cleanup EXIT

# Keep script running
wait $BACKEND_PID
