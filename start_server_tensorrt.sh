#!/bin/bash
# CozyVoice3 Server with TensorRT Acceleration

# Kill existing server
pkill -f "uvicorn.*openai_tts_cosyvoice_server" 2>/dev/null || true
sleep 2

# Set environment variables
export COSYVOICE_USE_VLLM=false
export COSYVOICE_USE_TRT=true
export COSYVOICE_FP16=true

# Start server
echo "Starting CozyVoice3 server with TensorRT..."
nohup conda run -n cosyvoice3 uvicorn openai_tts_cosyvoice_server:app \
    --host 0.0.0.0 \
    --port 8000 \
    > server_tensorrt.log 2>&1 &

echo "Server starting in background..."
echo "Log file: server_tensorrt.log"
echo ""
echo "Waiting for server to be ready..."
sleep 10

# Check if server is running
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ Server is ready!"
        curl -s http://localhost:8000/health | python -m json.tool
        exit 0
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

echo "✗ Server failed to start. Check server_tensorrt.log"
exit 1
