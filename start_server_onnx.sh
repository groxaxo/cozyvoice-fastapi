#!/bin/bash
# CozyVoice3 Server with ONNX Acceleration (Default Configuration)
#
# This script demonstrates running the server with ONNX-optimized 
# Flow and HiFi-GAN modules for improved inference performance.
# ONNX is enabled by default, but this script makes it explicit.

# Kill existing server
pkill -f "uvicorn.*openai_tts_cosyvoice_server" 2>/dev/null || true
sleep 2

# Set environment variables
# ONNX is enabled by default, but we set it explicitly here for clarity
export COSYVOICE_USE_ONNX=true
export COSYVOICE_ONNX_REPO=Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX

# Optional: Enable FP16 for faster inference
# export COSYVOICE_FP16=true

# Optional: Use a custom ONNX repository
# export COSYVOICE_ONNX_REPO=your-username/your-onnx-repo

# Start server
echo "Starting CozyVoice3 server with ONNX acceleration..."
echo "ONNX Repository: $COSYVOICE_ONNX_REPO"
echo ""

nohup conda run -n cosyvoice3 uvicorn openai_tts_cosyvoice_server:app \
    --host 0.0.0.0 \
    --port 8092 \
    > server_onnx.log 2>&1 &

echo "Server starting in background..."
echo "Log file: server_onnx.log"
echo ""
echo "Waiting for server to be ready..."
sleep 10

# Check if server is running
for i in {1..30}; do
    if curl -s http://localhost:8092/health > /dev/null 2>&1; then
        echo "✓ Server is ready!"
        echo ""
        echo "Health Check Response:"
        curl -s http://localhost:8092/health | python -m json.tool
        echo ""
        echo "ONNX models will be automatically downloaded if not present."
        echo "Check server_onnx.log for download progress."
        echo ""
        echo "To disable ONNX and use PyTorch, run:"
        echo "  export COSYVOICE_USE_ONNX=false"
        echo ""
        echo "Server is ready at http://localhost:8092"
        exit 0
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

echo "✗ Server failed to start. Check server_onnx.log for details"
tail -20 server_onnx.log
exit 1
