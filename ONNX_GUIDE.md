# ONNX Support Guide

This guide explains how to use ONNX-optimized models with the CosyVoice FastAPI server for improved inference performance.

## Overview

The server now supports ONNX (Open Neural Network Exchange) optimized versions of the Flow and HiFi-GAN modules. ONNX provides:

- **Better Performance**: Optimized inference compared to standard PyTorch
- **Easy Setup**: Automatic model download from Hugging Face
- **Broad Compatibility**: Works on CPU and GPU
- **Flexible Configuration**: Easy to enable/disable via environment variables

## Quick Start

### Default Configuration (ONNX Enabled)

ONNX support is **enabled by default**. Simply start the server normally:

```bash
conda activate cosyvoice3
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

The server will automatically:
1. Check for ONNX models in the model directory
2. Download them from Hugging Face if missing
3. Load ONNX-optimized Flow and HiFi-GAN modules

### Using the ONNX Startup Script

For convenience, use the provided startup script:

```bash
chmod +x start_server_onnx.sh
./start_server_onnx.sh
```

This script:
- Sets ONNX environment variables explicitly
- Starts the server with ONNX enabled
- Checks server health
- Displays helpful status information

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COSYVOICE_USE_ONNX` | `true` | Enable/disable ONNX modules |
| `COSYVOICE_ONNX_REPO` | `Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX` | Hugging Face repo for ONNX models |
| `COSYVOICE_FP16` | `false` | Use FP16 precision (affects which Flow model is loaded) |

### Configuration Examples

**1. Default ONNX (Recommended)**
```bash
# ONNX enabled with FP32 precision
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

**2. ONNX with FP16**
```bash
# ONNX enabled with FP16 precision for faster inference
export COSYVOICE_FP16=true
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

**3. Disable ONNX**
```bash
# Use original PyTorch implementation
export COSYVOICE_USE_ONNX=false
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

**4. Custom ONNX Repository**
```bash
# Use a different Hugging Face repository for ONNX models
export COSYVOICE_USE_ONNX=true
export COSYVOICE_ONNX_REPO=your-username/custom-onnx-repo
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

## ONNX Models

### Model Files

The ONNX implementation uses three model files:

1. **Flow Model (FP32)**: `flow_fp32.onnx`
   - Used when `COSYVOICE_FP16=false` (default)
   - Standard precision
   - Better quality, slightly slower

2. **Flow Model (FP16)**: `flow_fp16.onnx`
   - Used when `COSYVOICE_FP16=true`
   - Half precision
   - Faster inference, lower memory usage

3. **HiFi-GAN Vocoder**: `hift.onnx`
   - High-quality audio generation
   - Used in both FP32 and FP16 modes

### Model Location

ONNX models are stored in:
```
CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512/
├── flow_fp32.onnx
├── flow_fp16.onnx
└── hift.onnx
```

### Default Repository

The default Hugging Face repository for ONNX models is:
- **Repository**: `Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX`
- **URL**: https://huggingface.co/Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX

This repository contains:
- Separate ONNX files for Flow (FP32 and FP16) and HiFi-GAN
- Combined versions of the models
- Models optimized for CosyVoice 3.0 architecture

## Installation

### Automatic Installation

The server automatically handles ONNX model download:

```bash
# Start server - ONNX models will be downloaded automatically
conda activate cosyvoice3
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

Server log output:
```
INFO: Loading CosyVoice model...
INFO: Configuration: vLLM=False, TensorRT=False, FP16=False, Quantization=False, ONNX=True
INFO: Downloading ONNX models (flow_fp32.onnx, hift.onnx) from Hugging Face repo Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX...
INFO: ONNX models downloaded successfully.
INFO: Model loaded successfully
```

### Manual Installation

If you prefer to download models manually:

```bash
conda activate cosyvoice3
pip install huggingface_hub

# Create model directory if it doesn't exist
mkdir -p CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512

# Download ONNX models
python << EOF
from huggingface_hub import hf_hub_download

model_dir = 'CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512'
repo = 'Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX'

# Download FP32 Flow model
print("Downloading flow_fp32.onnx...")
hf_hub_download(
    repo_id=repo,
    filename='flow_fp32.onnx',
    local_dir=model_dir
)

# Download FP16 Flow model (optional)
print("Downloading flow_fp16.onnx...")
hf_hub_download(
    repo_id=repo,
    filename='flow_fp16.onnx',
    local_dir=model_dir
)

# Download HiFi-GAN model
print("Downloading hift.onnx...")
hf_hub_download(
    repo_id=repo,
    filename='hift.onnx',
    local_dir=model_dir
)

print("All ONNX models downloaded successfully!")
EOF
```

### Prerequisites

**Required**:
- `huggingface_hub` Python package (included in requirements.txt)
- ONNX Runtime (included with CosyVoice dependencies)

**Recommended for GPU acceleration**:
- cuDNN 8.9.7.29 or later

Install cuDNN:
```bash
conda activate cosyvoice3
conda install -c conda-forge cudnn=8.9.7.29 -y
```

## Verification

### 1. Check Server Health

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

### 2. Check Server Logs

Look for ONNX-related messages:
```
INFO: Configuration: vLLM=False, TensorRT=False, FP16=False, Quantization=False, ONNX=True
INFO: ONNX flow/hift models found in model directory.
INFO: Model loaded successfully
```

### 3. Check Web Interface

Visit `http://localhost:8092/` in your browser. The page header will display:
```
Backend: PyTorch | Voices: X | Model: Fun-CosyVoice3-0.5B-2512 | ONNX enabled
```

### 4. Test Speech Generation

```bash
curl http://localhost:8092/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Testing ONNX acceleration with CosyVoice.",
    "voice": "en"
  }' \
  --output test_onnx.wav
```

## Combining ONNX with Other Features

### ONNX + FP16

```bash
export COSYVOICE_USE_ONNX=true
export COSYVOICE_FP16=true
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

### ONNX + vLLM

```bash
export COSYVOICE_USE_ONNX=true
export COSYVOICE_USE_VLLM=true
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

### ONNX + Quantization

```bash
export COSYVOICE_USE_ONNX=true
export QUANTIZATION_ENABLED=true
export QUANTIZATION_BITS=4
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

### Note on TensorRT

When using TensorRT (`COSYVOICE_USE_TRT=true`), the TensorRT engine handles the Flow model optimization separately. ONNX support for HiFi-GAN can still be used alongside TensorRT.

## Troubleshooting

### ONNX Models Not Downloading

**Issue**: Models fail to download automatically

**Solutions**:
```bash
# 1. Check internet connectivity
ping huggingface.co

# 2. Verify huggingface_hub is installed
pip show huggingface_hub

# 3. Install if missing
pip install huggingface_hub

# 4. Manually download (see Manual Installation section)
```

### Wrong Precision Model Loaded

**Issue**: Server loads FP32 when you want FP16 (or vice versa)

**Solution**:
```bash
# For FP32
export COSYVOICE_FP16=false
export COSYVOICE_USE_ONNX=true

# For FP16
export COSYVOICE_FP16=true
export COSYVOICE_USE_ONNX=true

# Restart server
```

### ONNX Runtime Errors

**Issue**: ONNX Runtime errors during inference

**Solutions**:
```bash
# 1. Install/Update cuDNN for GPU acceleration
conda install -c conda-forge cudnn=8.9.7.29 -y

# 2. Verify ONNX Runtime installation
python -c "import onnxruntime; print(onnxruntime.__version__)"

# 3. Check GPU availability
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# 4. Disable ONNX if issues persist
export COSYVOICE_USE_ONNX=false
```

### Performance Issues

**Issue**: ONNX doesn't seem faster than PyTorch

**Solutions**:
```bash
# 1. Enable FP16 for better performance
export COSYVOICE_FP16=true
export COSYVOICE_USE_ONNX=true

# 2. Ensure cuDNN is installed for GPU acceleration
conda install -c conda-forge cudnn=8.9.7.29 -y

# 3. Check GPU is being used
nvidia-smi

# 4. Compare with ONNX disabled
export COSYVOICE_USE_ONNX=false
# Run benchmark, then re-enable ONNX and compare
```

### Download Fails Behind Proxy

**Issue**: Cannot download from Hugging Face due to proxy/firewall

**Solution**:
```bash
# Set proxy environment variables
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# Or download manually from another machine and copy files
```

### Custom Repository Not Working

**Issue**: Custom ONNX repository doesn't work

**Verification**:
```bash
# 1. Verify repository exists and is public
curl -I https://huggingface.co/your-username/your-onnx-repo

# 2. Check repository contains required files
# - flow_fp32.onnx or flow_fp16.onnx
# - hift.onnx

# 3. Test manual download
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='your-username/your-onnx-repo',
    filename='flow_fp32.onnx',
    local_dir='/tmp/test'
)
"
```

## Performance Considerations

### Expected Performance

ONNX provides moderate performance improvements over standard PyTorch:
- Faster inference for Flow and HiFi-GAN modules
- Lower memory usage with FP16
- Better CPU performance (useful for non-GPU deployments)

### Optimization Tips

1. **Use FP16 for faster inference**:
   ```bash
   export COSYVOICE_FP16=true
   ```

2. **Ensure cuDNN is installed** for GPU acceleration:
   ```bash
   conda install -c conda-forge cudnn=8.9.7.29 -y
   ```

3. **Monitor GPU usage**:
   ```bash
   nvidia-smi dmon -s u
   ```

4. **Compare configurations**:
   - Benchmark with ONNX enabled
   - Benchmark with ONNX disabled
   - Choose based on your hardware and requirements

### When to Use ONNX vs TensorRT

**Use ONNX when**:
- You want easy setup with good performance
- You're testing or prototyping
- You need broad hardware compatibility
- Setup time is limited

**Use TensorRT when**:
- You need maximum performance
- You have an NVIDIA GPU
- You can invest time in setup and optimization
- You're deploying to production

**Use Both**:
- ONNX for HiFi-GAN
- TensorRT for Flow model
- Best of both worlds (requires TensorRT setup)

## FAQ

**Q: Is ONNX enabled by default?**
A: Yes, `COSYVOICE_USE_ONNX=true` is the default setting.

**Q: Do I need to download ONNX models manually?**
A: No, the server automatically downloads them from Hugging Face on first run.

**Q: Can I use my own ONNX models?**
A: Yes, set `COSYVOICE_ONNX_REPO` to your Hugging Face repository, or manually place ONNX files in the model directory.

**Q: Does ONNX work with vLLM or TensorRT?**
A: Yes, ONNX can be combined with other acceleration options.

**Q: How do I disable ONNX?**
A: Set `COSYVOICE_USE_ONNX=false` before starting the server.

**Q: What if automatic download fails?**
A: See the Manual Installation section to download models manually using `huggingface_hub`.

**Q: Which precision should I use?**
A: FP32 for best quality, FP16 for faster inference. Start with FP32 and switch to FP16 if you need more speed.

**Q: Can ONNX run on CPU?**
A: Yes, ONNX Runtime supports CPU inference, though GPU is recommended for real-time performance.

## Additional Resources

- **CosyVoice Repository**: https://github.com/FunAudioLLM/CosyVoice
- **ONNX Models Repository**: https://huggingface.co/Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX
- **ONNX Runtime Documentation**: https://onnxruntime.ai/docs/
- **Hugging Face Hub**: https://huggingface.co/docs/huggingface_hub

## Support

For issues related to ONNX support:
1. Check this guide's Troubleshooting section
2. Review server logs for error messages
3. Verify ONNX models are present in the model directory
4. Test with ONNX disabled to isolate the issue
5. Open an issue on GitHub with logs and configuration details
