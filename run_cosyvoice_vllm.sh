#!/bin/bash
# Launch CosyVoice FastAPI with vLLM acceleration
# This script uses the official CosyVoice vLLM backend

set -e

echo "🚀 Starting CosyVoice FastAPI with vLLM acceleration..."

# Configuration
export COSYVOICE_USE_VLLM="true"
export COSYVOICE_USE_TRT="false"
export COSYVOICE_FP16="false"  # CosyVoice3 uses fp32 for vLLM
export TTS_API_KEY="not-needed"
PORT=8001  # Different port to avoid conflicts with standard server

# Conda environment with vLLM
ENV_NAME="cosyvoice3_vllm"

echo "📦 Using conda environment: $ENV_NAME"
echo "🔧 Settings: vLLM=true, TRT=false, FP16=false"
echo "🌐 Server will run on port: $PORT"
echo ""

# Check if environment exists
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "❌ Error: Conda environment '$ENV_NAME' not found!"
    echo "   Please run: conda create -n cosyvoice3_vllm --clone cosyvoice3"
    echo "   Then install: conda run -n $ENV_NAME pip install vllm==v0.9.0"
    exit 1
fi

# Check if already running on this port
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "❌ Error: Port $PORT is already in use!"
    echo "   Run './cleanup_servers.sh' to stop existing servers first."
    exit 1
fi

# Check GPU availability
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ Error: No NVIDIA GPU detected!"
    exit 1
fi

# Verify GPU 2 is available if CUDA_VISIBLE_DEVICES is set
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "🎮 Using GPU(s): $CUDA_VISIBLE_DEVICES"
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting server..."
echo "📊 Press Ctrl+C to stop the server"
echo ""

# Launch server (no auto-restart - use systemd/supervisor for that)
# exec replaces the shell with uvicorn, ensuring clean shutdown on SIGTERM
exec conda run -n $ENV_NAME uvicorn openai_tts_cosyvoice_server:app \
    --port $PORT \
    --host 0.0.0.0 \
    --workers 1 \
    --log-level info
