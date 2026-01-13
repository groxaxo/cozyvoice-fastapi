# Installation Guide - CozyVoice FastAPI Server

This guide provides comprehensive installation instructions for the CozyVoice FastAPI server with optional TensorRT acceleration.

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA 11.8+ (Required for real-time performance)
- **Memory**: 16GB+ RAM recommended
- **Storage**: 10GB+ free space for models and dependencies

### Software Requirements

1. **CUDA Toolkit** (11.8 or higher)
   ```bash
   # Verify CUDA installation
   nvidia-smi
   nvcc --version
   ```

2. **Conda** (Miniconda or Anaconda)
   ```bash
   # Install Miniconda if not already installed
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

3. **Git**
   ```bash
   sudo apt-get update
   sudo apt-get install git -y
   ```

---

## Quick Start Installation

### 1. Clone the Repository

```bash
git clone https://github.com/groxaxo/cozyvoice-fastapi.git
cd cozyvoice-fastapi
```

### 2. Automated Setup

The easiest way to get started is using the autonomous setup script:

```bash
chmod +x run_cosyvoice_autonomous.sh
./run_cosyvoice_autonomous.sh
```

This script will:
- Create a `cosyvoice3` conda environment
- Install all required dependencies
- Download the CosyVoice3-0.5B model
- Install cuDNN for ONNX Runtime support
- Start the server on port 8092

**Note**: The first run may take 10-15 minutes to download models and install dependencies.

---

## Manual Installation

If you prefer manual setup or need more control:

### 1. Create Conda Environment

```bash
conda create -n cosyvoice3 python=3.10 -y
conda activate cosyvoice3
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install cuDNN for ONNX Runtime support
conda install -c conda-forge cudnn=8.9.7.29 -y
```

### 3. Clone CosyVoice Repository

```bash
# Clone the CosyVoice repository (if not already present)
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive
cd ..
```

### 4. Download Models

The models will be automatically downloaded on first run, or you can manually download:

```bash
# Models are downloaded to CosyVoice/pretrained_models/
# The server will handle this automatically
```

### 5. Verify Installation

```bash
conda activate cosyvoice3
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## ONNX Support (Enabled by Default)

The server now supports ONNX-optimized Flow and HiFi-GAN modules for improved inference performance. This feature is **enabled by default** and provides a good balance between performance and ease of setup.

### Overview

- **Status**: Enabled by default (`COSYVOICE_USE_ONNX=true`)
- **Models**: Optimized ONNX versions of Flow and HiFi-GAN modules
- **Source**: Automatically downloaded from Hugging Face (`Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX`)
- **Compatibility**: Works alongside vLLM, TensorRT, and quantization options

### Quick Start with ONNX

**1. Default Configuration (ONNX Enabled)**

The server automatically enables ONNX support. Just start normally:

```bash
conda activate cosyvoice3
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

On first run, the server will automatically download the ONNX models if they're not present.

**2. Disable ONNX (Use PyTorch)**

If you prefer the original PyTorch implementation:

```bash
export COSYVOICE_USE_ONNX=false
conda activate cosyvoice3
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

**3. Use Custom ONNX Repository**

To use a different Hugging Face repository for ONNX models:

```bash
export COSYVOICE_USE_ONNX=true
export COSYVOICE_ONNX_REPO=your-username/custom-onnx-repo
conda activate cosyvoice3
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

### ONNX Models Details

The ONNX implementation uses the following models:

**Flow Model**:
- `flow_fp32.onnx` - Standard precision (used when `COSYVOICE_FP16=false`)
- `flow_fp16.onnx` - Half precision (used when `COSYVOICE_FP16=true`)

**HiFi-GAN Model**:
- `hift.onnx` - Vocoder model for high-quality audio generation

**Default Repository**: `Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX`

**Model Directory**: `CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512/`

### Automatic Download

The server automatically downloads ONNX models on first run if they're not present:

```
INFO: Loading CosyVoice model...
INFO: Configuration: vLLM=False, TensorRT=False, FP16=False, Quantization=False, ONNX=True
INFO: Downloading ONNX models (flow_fp32.onnx, hift.onnx) from Hugging Face repo Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX...
INFO: ONNX models downloaded successfully.
INFO: Model loaded successfully
```

### Manual Download (Optional)

If automatic download fails or you want to pre-download the models:

```bash
conda activate cosyvoice3
pip install huggingface_hub

# Download FP32 models
python -c "
from huggingface_hub import hf_hub_download
model_dir = 'CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512'
repo = 'Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX'
hf_hub_download(repo_id=repo, filename='flow_fp32.onnx', local_dir=model_dir)
hf_hub_download(repo_id=repo, filename='hift.onnx', local_dir=model_dir)
print('ONNX models downloaded successfully!')
"

# For FP16 (optional, only if using COSYVOICE_FP16=true)
python -c "
from huggingface_hub import hf_hub_download
model_dir = 'CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512'
repo = 'Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX'
hf_hub_download(repo_id=repo, filename='flow_fp16.onnx', local_dir=model_dir)
print('FP16 ONNX model downloaded successfully!')
"
```

### Verifying ONNX Installation

**1. Check Health Endpoint**

```bash
curl http://localhost:8092/health
```

Expected response with ONNX enabled:
```json
{
  "status": "ok",
  "model": "cosyvoice3",
  "backend": "pytorch",
  "onnx": true
}
```

**2. Check Server Logs**

Look for ONNX-related messages in the server output:
```
INFO: Configuration: vLLM=False, TensorRT=False, FP16=False, Quantization=False, ONNX=True
INFO: ONNX flow/hift models found in model directory.
```

**3. Check Web Interface**

Visit `http://localhost:8092/` in your browser. The ONNX status will be displayed as a badge in the page header.

### ONNX Requirements

**Python Packages**:
- `huggingface_hub` - For automatic model download (installed via requirements.txt)
- ONNX Runtime (included with CosyVoice dependencies)

**Note**: cuDNN is recommended for optimal ONNX Runtime performance on NVIDIA GPUs:
```bash
conda install -c conda-forge cudnn=8.9.7.29 -y
```

### Combining ONNX with Other Options

ONNX can be combined with other acceleration options:

**ONNX + FP16**:
```bash
export COSYVOICE_USE_ONNX=true
export COSYVOICE_FP16=true
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

**ONNX + vLLM**:
```bash
export COSYVOICE_USE_ONNX=true
export COSYVOICE_USE_VLLM=true
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

**Note**: When using TensorRT (`COSYVOICE_USE_TRT=true`), ONNX support for the Flow model is handled differently as TensorRT uses its own ONNX conversion.

### Troubleshooting ONNX

**ONNX Models Not Downloading**

```bash
# Check internet connectivity to Hugging Face
ping huggingface.co

# Verify huggingface_hub is installed
pip show huggingface_hub

# Manually download models (see Manual Download section above)
```

**ONNX Runtime Errors**

```bash
# Ensure cuDNN is installed for GPU acceleration
conda install -c conda-forge cudnn=8.9.7.29 -y

# Check ONNX Runtime is available
python -c "import onnxruntime; print(onnxruntime.__version__)"
```

**Performance Issues with ONNX**

```bash
# Try disabling ONNX to compare performance
export COSYVOICE_USE_ONNX=false

# Or try FP16 for faster inference
export COSYVOICE_FP16=true
export COSYVOICE_USE_ONNX=true
```

**Wrong Precision Model**

If the server is loading the wrong precision ONNX model:
```bash
# For FP32
export COSYVOICE_FP16=false
export COSYVOICE_USE_ONNX=true

# For FP16
export COSYVOICE_FP16=true
export COSYVOICE_USE_ONNX=true
```

---

## TensorRT Installation (Optional - For Maximum Performance)

TensorRT provides 2-3x faster inference for the Flow model. Follow these steps to enable it:

### Prerequisites for TensorRT

- NVIDIA GPU with Compute Capability 7.0+ (RTX 2060 or newer)
- CUDA 11.8 or higher
- 8GB+ GPU memory

### Automated TensorRT Setup

```bash
chmod +x setup_tensorrt.sh
./setup_tensorrt.sh
```

This script will:
1. Check prerequisites (CUDA, conda environment)
2. Install TensorRT if not present
3. Export the Flow model to ONNX format (~2-3 minutes)
4. Generate TensorRT engine (~5-10 minutes)
5. Create startup script for TensorRT-enabled server

**Expected output:**
- `flow.decoder.estimator.fp32.onnx` (1.3GB)
- `flow.decoder.estimator.fp16.mygpu.plan` (637MB)
- `start_server_tensorrt.sh` (startup script)

### Manual TensorRT Setup

If you prefer manual installation:

```bash
# 1. Install TensorRT
conda activate cosyvoice3
pip install tensorrt

# 2. Export ONNX model
cd CosyVoice
python cosyvoice/bin/export_onnx.py --model_dir ../CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512
cd ..

# 3. Generate TensorRT engine
python -c "
import sys
sys.path.insert(0, 'CosyVoice')
sys.path.insert(0, 'CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice3
cosyvoice = CosyVoice3('CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512', load_trt=True, fp16=True)
print('TensorRT engine generated successfully!')
"
```

---

## Starting the Server

### Standard Server (PyTorch with ONNX - Default)

```bash
conda activate cosyvoice3
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

**Note**: ONNX is enabled by default for optimized performance.

### PyTorch Only (ONNX Disabled)

```bash
export COSYVOICE_USE_ONNX=false
conda activate cosyvoice3
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

### With ONNX + FP16

```bash
export COSYVOICE_USE_ONNX=true
export COSYVOICE_FP16=true
conda activate cosyvoice3
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

### With vLLM Acceleration

```bash
export COSYVOICE_USE_VLLM=true
export COSYVOICE_FP16=true
conda run -n cosyvoice3 uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

### With TensorRT Acceleration (Fastest)

```bash
# Using the startup script
./start_server_tensorrt.sh

# Or manually
export COSYVOICE_USE_TRT=true
export COSYVOICE_FP16=true
conda run -n cosyvoice3 uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000
```

---

## Verification

### 1. Check Server Health

```bash
curl http://localhost:8092/health
```

Expected response:
```json
{
  "status": "healthy",
  "model": "CosyVoice3-0.5B",
  "backend": "pytorch",
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 3090"
}
```

### 2. Test Speech Generation

```bash
curl http://localhost:8092/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Hello! This is a test of the CosyVoice TTS server.",
    "voice": "aimee-en"
  }' \
  --output test_output.wav
```

### 3. List Available Voices

```bash
curl http://localhost:8092/v1/voices
```

---

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:

```bash
# Reduce GPU memory usage
export VLLM_GPU_MEMORY_UTILIZATION=0.3

# Or use CPU fallback for testing
export CUDA_VISIBLE_DEVICES=""
```

### cuDNN Not Found

```bash
# Reinstall cuDNN
conda activate cosyvoice3
conda install -c conda-forge cudnn=8.9.7.29 -y
```

### Port Already in Use

```bash
# Kill existing server
pkill -f "uvicorn.*openai_tts_cosyvoice_server"

# Or use a different port
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8093
```

### TensorRT Engine Generation Fails

```bash
# Check TensorRT installation
python -c "import tensorrt; print(tensorrt.__version__)"

# Verify ONNX file exists
ls -lh CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512/flow.decoder.estimator.fp32.onnx

# Check logs
tail -f tensorrt_setup.log
```

### Model Download Issues

If models fail to download automatically:

```bash
# Models are hosted on ModelScope/HuggingFace
# The server will attempt to download on first run
# Check server logs for download progress
tail -f server.log
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COSYVOICE_USE_ONNX` | `true` | Enable ONNX-optimized Flow/HiFi-GAN modules |
| `COSYVOICE_ONNX_REPO` | `Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX` | Hugging Face repository for ONNX models |
| `COSYVOICE_USE_VLLM` | `false` | Enable vLLM acceleration for LLM |
| `COSYVOICE_USE_TRT` | `false` | Enable TensorRT acceleration for Flow model |
| `COSYVOICE_FP16` | `false` | Use FP16 precision (faster, less memory) |
| `QUANTIZATION_ENABLED` | `false` | Enable BitsAndBytes quantization |
| `QUANTIZATION_BITS` | `4` | Quantization bits (4 or 8) |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory fraction for vLLM |
| `CUDA_VISIBLE_DEVICES` | All GPUs | Specify which GPUs to use (e.g., `0,1`) |
| `TTS_API_KEY` | `not-needed` | API key for authentication (optional) |

---

## Next Steps

- See [README.md](README.md) for API usage examples
- See [PERFORMANCE.md](PERFORMANCE.md) for optimization tips
- Add custom voices to `voice_samples/` directory
- Configure for production deployment

---

## Support

For issues and questions:
- Check the [Troubleshooting](#troubleshooting) section
- Review server logs: `tail -f server.log`
- Open an issue on GitHub
