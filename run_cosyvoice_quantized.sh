#!/bin/bash

# CosyVoice FastAPI Server with 4-bit Quantization
# This script runs the CosyVoice TTS server with BitsAndBytes 4-bit quantization
# to reduce VRAM usage by ~75% (2GB → 500MB)

set -e

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Check if the conda environment exists
if ! conda env list | grep -q "cosyvoice3"; then
    echo "Error: cosyvoice3 conda environment not found"
    echo "Please create it first with: conda create -n cosyvoice3 python=3.10 -y"
    exit 1
fi

# Check if port 8000 is available
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Error: Port 8000 is already in use"
    echo "Please stop the existing server or use a different port"
    exit 1
fi

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

echo "Starting CosyVoice TTS server with 4-bit quantization..."
echo "VRAM usage will be reduced by ~75% (2GB → 500MB)"
echo ""
echo "Configuration:"
echo "  - Backend: PyTorch"
echo "  - Quantization: 4-bit (NF4)"
echo "  - Port: 8000"
echo ""

# Set environment variables for 4-bit quantization
export TTS_API_KEY="${TTS_API_KEY:-not-needed}"
export QUANTIZATION_ENABLED="true"
export QUANTIZATION_BITS="4"

# Activate conda environment and run server
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cosyvoice3

echo "Installing quantization dependencies if needed..."
pip install -q bitsandbytes>=0.41.0 transformers>=4.48.0 accelerate>=0.20.0 2>/dev/null || true

echo ""
echo "Server starting on http://0.0.0.0:8000"
echo "Press Ctrl+C to stop"
echo ""

# Run the server
exec uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000
