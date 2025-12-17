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

# Launch server with auto-restart
while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting server..."
    
    conda run -n $ENV_NAME python openai_tts_cosyvoice_server.py \
        --port $PORT \
        --host 0.0.0.0 \
        || {
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Server crashed! Restarting in 5 seconds..."
            sleep 5
        }
done
