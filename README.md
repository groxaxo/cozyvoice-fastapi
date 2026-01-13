# CosyVoice FastAPI Server

A robust, production-ready FastAPI server for **CosyVoice 3**, offering an OpenAI-compatible TTS endpoint. This project enables zero-shot voice cloning, multilingual speech synthesis, and seamless integration with platforms like Open-WebUI.

## 🚀 Key Features

*   **OpenAI-Compatible API**: Drop-in replacement for OpenAI's `/v1/audio/speech` and `/v1/models` endpoints.
*   **Zero-Shot Voice Cloning**: Clone voices from short audio samples (3s+).
*   **Multilingual Support**: Supports English, Spanish, French, Chinese, Japanese, Korean, and more.
*   **Advanced Voice Management**: Automatic voice discovery from the `voice_samples/` directory with language detection.
*   **High Performance**: Multiple acceleration options available:
    *   **PyTorch (Baseline)**: RTF 0.364 - Stable and reliable
    *   **vLLM + FP16**: RTF 0.362 - Easy setup, production-ready
    *   **TensorRT + FP16**: RTF 0.340 - Maximum performance (6.6% faster)
*   **Production Ready**: Includes text normalization, smart sentence splitting, and stable concurrency handling.

📚 **Documentation**: [Installation Guide](INSTALLATION.md) | [Performance Benchmarks](PERFORMANCE.md) | [ONNX Guide](ONNX_GUIDE.md)

---

## 🛠️ Setup & Installation

### Prerequisites
*   **OS**: Linux (Ubuntu 20.04+ recommended)
*   **GPU**: NVIDIA GPU with CUDA 11.8+ (Recommended for real-time performance)
*   **Conda**: Miniconda or Anaconda installed

### Quick Start

1.  **Clone the Repository**
    ```bash
    git clone <repository_url>
    cd cozyvoice_fastapi
    ```

2.  **Run the Setup Script**
    This script initializes the environment, installs dependencies, and downloads necessary models.
    ```bash
    chmod +x run_cosyvoice_autonomous.sh
    ./run_cosyvoice_autonomous.sh
    ```
    *This will create a `cosyvoice3` conda environment and start the server on port 8092.*

### Manual Installation
If you prefer manual setup:
```bash
conda create -n cosyvoice3 python=3.10 -y
conda activate cosyvoice3
pip install -r requirements.txt
# Additional model downloads may be required
# IMPORTANT: Install CUDNN for ONNX Runtime support
conda install -c conda-forge cudnn=8.9.7.29 -y
```

---

## ⚡ TensorRT Setup (Optional - For Maximum Performance)

TensorRT provides up to 6.6% performance improvement over baseline. For detailed instructions, see [INSTALLATION.md](INSTALLATION.md#tensorrt-installation-optional---for-maximum-performance).

### Quick TensorRT Setup

```bash
# Automated setup (recommended)
chmod +x setup_tensorrt.sh
./setup_tensorrt.sh

# This will:
# 1. Install TensorRT
# 2. Export Flow model to ONNX (~2-3 minutes)
# 3. Generate TensorRT engine (~5-10 minutes)
# 4. Create startup script
```

### Start Server with TensorRT

```bash
# Using the startup script
./start_server_tensorrt.sh

# Or manually
export COSYVOICE_USE_TRT=true
export COSYVOICE_FP16=true
conda run -n cosyvoice3 uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000
```

### Performance Comparison

| Configuration | RTF | Speedup | Setup Complexity |
|--------------|-----|---------|------------------|
| PyTorch (Baseline) | 0.364 | 1.00x | Easy |
| vLLM + FP16 | 0.362 | 1.01x | Easy |
| **TensorRT + FP16** | **0.340** | **1.07x** | Moderate |

*RTF = Real-Time Factor (lower is better). See [PERFORMANCE.md](PERFORMANCE.md) for detailed benchmarks.*

---

## 🎯 ONNX Support (Optional - Alternative Acceleration)

The server now supports using ONNX-optimized Flow and HiFi-GAN modules for faster inference as an alternative to PyTorch. This feature is **enabled by default** and provides a performance boost without the complexity of TensorRT setup.

### Key Features

*   **Easy Setup**: ONNX models are automatically downloaded from Hugging Face when needed
*   **Default Enabled**: Works out-of-the-box with `COSYVOICE_USE_ONNX=true` (default)
*   **Flexible**: Easy to disable and fall back to PyTorch if needed
*   **Compatible**: Works alongside other acceleration options (vLLM, TensorRT, quantization)

### Configuration

The ONNX feature is controlled by two environment variables:

```bash
# Enable/disable ONNX (default: true)
export COSYVOICE_USE_ONNX=true

# Specify the Hugging Face repository for ONNX models (default: Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX)
export COSYVOICE_ONNX_REPO=Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX
```

### Quick Start with ONNX

**Option 1: Use Default ONNX (Recommended)**
```bash
# ONNX is enabled by default, just start the server normally
conda activate cosyvoice3
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

**Option 2: Explicitly Enable ONNX**
```bash
# Explicitly enable ONNX
export COSYVOICE_USE_ONNX=true
conda activate cosyvoice3
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

**Option 3: Disable ONNX (Fall Back to PyTorch)**
```bash
# Disable ONNX to use original PyTorch implementation
export COSYVOICE_USE_ONNX=false
conda activate cosyvoice3
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

**Option 4: Use Custom ONNX Repository**
```bash
# Use a different Hugging Face repository for ONNX models
export COSYVOICE_USE_ONNX=true
export COSYVOICE_ONNX_REPO=your-username/your-onnx-repo
conda activate cosyvoice3
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

### ONNX Models

The ONNX implementation uses optimized versions of the Flow and HiFi-GAN modules:

*   **Flow Models**: `flow_fp32.onnx` (standard) or `flow_fp16.onnx` (when `COSYVOICE_FP16=true`)
*   **HiFi-GAN Model**: `hift.onnx`

These models are automatically downloaded from the specified Hugging Face repository (`Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX` by default) into your model directory (`CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512/`) when first needed.

**Manual Download** (optional):
```bash
# If automatic download fails, you can manually download ONNX files
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
model_dir = 'CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512'
repo = 'Lourdle/Fun-CosyVoice3-0.5B-2512_ONNX'
hf_hub_download(repo_id=repo, filename='flow_fp32.onnx', local_dir=model_dir)
hf_hub_download(repo_id=repo, filename='hift.onnx', local_dir=model_dir)
"
```

### Checking ONNX Status

You can verify if ONNX is enabled by checking the server's health endpoint:

```bash
curl http://localhost:8092/health
# Response includes: {"status": "ok", "model": "cosyvoice3", "backend": "pytorch", "onnx": true}
```

Or visit the web interface at `http://localhost:8092/` - the ONNX status will be displayed as a badge in the header.

### ONNX vs TensorRT

| Feature | ONNX | TensorRT |
|---------|------|----------|
| Setup Complexity | Very Easy (auto-download) | Moderate (requires compilation) |
| Performance Gain | Moderate | Higher |
| Compatibility | Broad (CPU/GPU) | NVIDIA GPUs only |
| First-time Setup | ~1 minute (download) | ~10-15 minutes (compilation) |
| Recommended For | Quick setup, testing | Maximum performance |

**Recommendation**: Start with ONNX (default) for easy setup and good performance. Switch to TensorRT if you need maximum speed and have time for the setup process.

---

## 🏃 Usage

### Starting the Server
The easiest way to run the server is using the autonomous runner:
```bash
./run_cosyvoice_autonomous.sh
```
Or manually:
```bash
conda activate cosyvoice3
conda run -n cosyvoice3 uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8092
```

### API Endpoints

#### 1. Generate Speech (`POST /v1/audio/speech`)
Generate audio from text.
```bash
curl http://localhost:8092/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Hello! This is a test of the CosyVoice TTS server.",
    "voice": "aimee-en"
  }' \
  --output output.wav
```

#### 2. List Voices (`GET /v1/voices`)
See all available voices detected in `voice_samples/`.
```bash
curl http://localhost:8092/v1/voices
```

#### 3. List Models (`GET /v1/models`)
Check available models.
```bash
curl http://localhost:8092/v1/models
```

---

## 🎤 Voice Management

### Adding Custom Voices
You can add your own voices for cloning by placing audio files in the `voice_samples/` directory.

1.  **Prepare Audio**: WAV (recommended), MP3, or FLAC. 3-10 seconds of clear speech.
2.  **Name File**: Use the format `[name]-[language_code].[ext]`.
    *   Example: `alice-en.wav` (English)
    *   Example: `carlos-es.mp3` (Spanish)
3.  **Place in Folder**: Move to `voice_samples/`.

The server automatically detects these files on startup and maps them.

### Naming Convention
The server uses the filename to infer the voice ID and language:

| Filename | Voice ID | Language |
| :--- | :--- | :--- |
| `aimee-en.wav` | `aimee-en` | English (`en`) |
| `lucho-es.wav` | `lucho-es` | Spanish (`es`) |
| `custom.wav` | `custom` | *Default/Auto* |

**Supported Language Codes**: `en`, `es`, `fr`, `ja`, `ko`, `zh`

---

## 🧩 Architecture

The system is built in layers to ensure stability and quality:

1.  **API Layer** (FastAPI): Handles requests, validation, and authentication.
2.  **Text Processing**: Normalizes text, handles numbers/symbols, and splits long sentences for streaming stability.
3.  **Voice Manager**: Scans `voice_samples/`, validates audio, and maintains a registry of available speakers.
4.  **Inference Engine**:
    *   **CosyVoice3-0.5B**: Main model for synthesis.
    *   **Flow Matching & HiFi-GAN**: High-fidelity audio generation.
    *   **GPU/CUDA**: Accelerated processing.

---

## ❓ Troubleshooting

### Common Issues

*   **"Model not found" error**: Ensure you are using `model="cosyvoice3"` or generic `tts-1` in your request.
*   **Audio is cut off**: Try shorter input text or check server logs for timeout issues.
*   **CUDA errors**: Verify `nvidia-smi` shows your GPU and that the correct torch version is installed.
*   **Port already in use**: Kill existing server with `pkill -f "uvicorn.*openai_tts_cosyvoice_server"` or use a different port.

### TensorRT-Specific Issues

*   **TensorRT engine generation fails**: 
    ```bash
    # Verify TensorRT installation
    conda activate cosyvoice3
    python -c "import tensorrt; print(tensorrt.__version__)"
    
    # Check ONNX file exists
    ls -lh CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512/flow.decoder.estimator.fp32.onnx
    ```

*   **Server shows "backend: pytorch" instead of "tensorrt"**:
    ```bash
    # Ensure environment variables are set
    export COSYVOICE_USE_TRT=true
    export COSYVOICE_FP16=true
    
    # Restart server
    ./start_server_tensorrt.sh
    ```

*   **CUDA Out of Memory with TensorRT**:
    ```bash
    # TensorRT engine uses ~800MB GPU memory
    # Ensure sufficient GPU memory available
    nvidia-smi
    ```

### Performance Issues

*   **Slow generation (RTF > 0.5)**:
    - Verify GPU is being used: Check server logs for "cuda" device
    - Try TensorRT for best performance: `./setup_tensorrt.sh`
    - Check GPU utilization: `nvidia-smi dmon -s u`

*   **Inconsistent performance**:
    - First request is slower (model loading)
    - Subsequent requests should be faster
    - Monitor GPU temperature and throttling

For detailed troubleshooting, see [INSTALLATION.md](INSTALLATION.md#troubleshooting).

## 🤝 Contributing
Feel free to open issues or submit PRs to improve performance or add features!
