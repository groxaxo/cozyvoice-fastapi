#!/bin/bash
# CosyVoice Server Cleanup Script
# Stops all running CosyVoice server instances cleanly

echo "🧹 Cleaning up CosyVoice server processes..."

# Find all uvicorn processes running our server
PIDS=$(pgrep -f "uvicorn openai_tts_cosyvoice_server")

if [ -z "$PIDS" ]; then
    echo "✅ No server processes found running"
    exit 0
fi

echo "Found processes: $PIDS"
echo "Sending SIGTERM for graceful shutdown..."

# Try graceful shutdown first
pkill -TERM -f "uvicorn openai_tts_cosyvoice_server"

# Wait up to 10 seconds for graceful shutdown
for i in {1..10}; do
    sleep 1
    if ! pgrep -f "uvicorn openai_tts_cosyvoice_server" > /dev/null; then
        echo "✅ All server processes stopped gracefully"
        exit 0
    fi
    echo -n "."
done

echo ""
echo "⚠️  Some processes still running, forcing shutdown..."
pkill -9 -f "uvicorn openai_tts_cosyvoice_server"

sleep 2

# Final verification
if pgrep -f "uvicorn openai_tts_cosyvoice_server" > /dev/null; then
    echo "❌ Failed to stop all processes. Please check manually with:"
    echo "   ps aux | grep uvicorn"
    exit 1
else
    echo "✅ All server processes stopped (forced)"
    exit 0
fi
