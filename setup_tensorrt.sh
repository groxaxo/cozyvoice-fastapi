#!/bin/bash
# TensorRT Auto-Installer for CozyVoice3 - Updated Version
# This script exports ONNX and generates TensorRT engine for the Flow model

set -e  # Exit on error

echo "=========================================="
echo "CozyVoice3 TensorRT Auto-Installer v2"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
CONDA_ENV="cosyvoice3"
MODEL_DIR="CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"
LOG_FILE="tensorrt_setup.log"

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Start logging
echo "Starting TensorRT setup at $(date)" > "$LOG_FILE"
echo ""

# Step 1: Check prerequisites
echo "Step 1: Checking prerequisites..."
echo "=================================="

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. CUDA is required for TensorRT."
    exit 1
fi
print_status "CUDA available"

# Check conda environment
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    print_error "Conda environment '${CONDA_ENV}' not found."
    exit 1
fi
print_status "Conda environment '${CONDA_ENV}' found"

# Check TensorRT installation
echo ""
echo "Checking TensorRT installation..."
if conda run -n "$CONDA_ENV" python -c "import tensorrt; print(f'TensorRT {tensorrt.__version__}')" 2>&1 | tee -a "$LOG_FILE"; then
    print_status "TensorRT already installed"
else
    print_warning "TensorRT not found. Installing..."
    conda run -n "$CONDA_ENV" pip install tensorrt 2>&1 | tee -a "$LOG_FILE"
    print_status "TensorRT installed"
fi

# Check PyTorch and CUDA
echo ""
echo "Checking PyTorch and CUDA versions..."
conda run -n "$CONDA_ENV" python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU Count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
" | tee -a "$LOG_FILE"

print_status "Prerequisites check complete"
echo ""

# Step 2: Check model files
echo "Step 2: Checking model files..."
echo "================================"

if [ ! -f "${MODEL_DIR}/flow.pt" ]; then
    print_error "Flow model not found at ${MODEL_DIR}/flow.pt"
    exit 1
fi
print_status "Flow model found"

if [ ! -f "${MODEL_DIR}/cosyvoice3.yaml" ]; then
    print_error "Config file not found at ${MODEL_DIR}/cosyvoice3.yaml"
    exit 1
fi
print_status "Config file found"

echo ""

# Step 3: Export ONNX model
echo "Step 3: Exporting ONNX model..."
echo "================================"

ONNX_FILE="${MODEL_DIR}/flow.decoder.estimator.fp32.onnx"

if [ -f "$ONNX_FILE" ]; then
    print_warning "ONNX model already exists"
    read -p "Do you want to regenerate it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Using existing ONNX model"
        SKIP_ONNX=true
    else
        print_status "Will regenerate ONNX model"
        SKIP_ONNX=false
        rm -f "$ONNX_FILE"
    fi
else
    SKIP_ONNX=false
fi

if [ "$SKIP_ONNX" = false ]; then
    print_warning "Exporting ONNX model (this may take 2-3 minutes)..."
    
    cd CosyVoice
    if conda run -n "$CONDA_ENV" python cosyvoice/bin/export_onnx.py \
        --model_dir "../${MODEL_DIR}" 2>&1 | tee -a "../$LOG_FILE"; then
        print_status "ONNX model exported successfully"
    else
        print_error "Failed to export ONNX model"
        print_warning "Check $LOG_FILE for details"
        cd ..
        exit 1
    fi
    cd ..
    
    if [ -f "$ONNX_FILE" ]; then
        ONNX_SIZE=$(du -h "$ONNX_FILE" | cut -f1)
        print_status "ONNX model created (${ONNX_SIZE})"
    else
        print_error "ONNX file not found after export"
        exit 1
    fi
fi

echo ""

# Step 4: Generate TensorRT engine
echo "Step 4: Generating TensorRT engine..."
echo "======================================"

TRT_FILE="${MODEL_DIR}/flow.decoder.estimator.fp16.mygpu.plan"

if [ -f "$TRT_FILE" ]; then
    print_warning "TensorRT engine already exists"
    read -p "Do you want to regenerate it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Using existing TensorRT engine"
        SKIP_TRT=true
    else
        print_status "Will regenerate TensorRT engine"
        SKIP_TRT=false
        rm -f "$TRT_FILE"
    fi
else
    SKIP_TRT=false
fi

if [ "$SKIP_TRT" = false ]; then
    print_warning "Converting ONNX to TensorRT (this may take 5-10 minutes)..."
    
    # Create a Python script to generate TRT engine
    cat > /tmp/generate_trt.py << 'PYTHON_SCRIPT'
import sys
import os
sys.path.insert(0, 'CosyVoice')
sys.path.insert(0, 'CosyVoice/third_party/Matcha-TTS')

import torch
from cosyvoice.cli.cosyvoice import CosyVoice3

print("Loading CosyVoice3 model...")
model_dir = "CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"

# Initialize model with TRT
print("Initializing with TensorRT...")
cosyvoice = CosyVoice3(model_dir, load_trt=True, fp16=True)

print("✓ TensorRT engine generated successfully!")
print(f"Engine saved to: {model_dir}/flow.decoder.estimator.fp16.mygpu.plan")
PYTHON_SCRIPT

    # Run the generation script
    if conda run -n "$CONDA_ENV" python /tmp/generate_trt.py 2>&1 | tee -a "$LOG_FILE"; then
        print_status "TensorRT engine generated successfully"
    else
        print_error "Failed to generate TensorRT engine"
        print_warning "Check $LOG_FILE for details"
        rm /tmp/generate_trt.py
        exit 1
    fi
    
    rm /tmp/generate_trt.py
    
    if [ -f "$TRT_FILE" ]; then
        TRT_SIZE=$(du -h "$TRT_FILE" | cut -f1)
        print_status "TensorRT engine created (${TRT_SIZE})"
    else
        print_error "TensorRT file not found after generation"
        exit 1
    fi
fi

echo ""

# Step 5: Create startup script
echo "Step 5: Creating startup script..."
echo "==================================="

cat > start_server_tensorrt.sh << 'STARTUP_SCRIPT'
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
STARTUP_SCRIPT

chmod +x start_server_tensorrt.sh
print_status "Startup script created: start_server_tensorrt.sh"

echo ""

# Step 6: Summary
echo "=========================================="
echo "TensorRT Setup Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - TensorRT version: $(conda run -n $CONDA_ENV python -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null)"
echo "  - ONNX model: $ONNX_FILE ($(du -h $ONNX_FILE 2>/dev/null | cut -f1))"
echo "  - TRT engine: $TRT_FILE ($(du -h $TRT_FILE 2>/dev/null | cut -f1))"
echo "  - Log file: $LOG_FILE"
echo ""
echo "To start the server with TensorRT:"
echo "  ./start_server_tensorrt.sh"
echo ""
echo "Or manually:"
echo "  export COSYVOICE_USE_TRT=true"
echo "  export COSYVOICE_FP16=true"
echo "  conda run -n cosyvoice3 uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000"
echo ""
echo "Expected performance improvement: 2-3x faster (RTF: 0.1-0.2)"
echo ""
print_status "Setup complete! Ready to test."
