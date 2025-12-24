# CosyVoice FastAPI Server

This is an OpenAI-compatible TTS server using CosyVoice3.

## Setup

1.  **Prerequisites**:
    *   Conda
    *   NVIDIA GPU with CUDA support

2.  **Installation**:
    The environment `cosyvoice3` has been created with all dependencies.

    To activate:
    ```bash
    conda activate cosyvoice3
    # Ensure compatible torchvision version
    pip install torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    ```

3.  **Running the Server**:
    ```bash
    # Default is "not-needed" (no auth)
    export TTS_API_KEY="not-needed" 
    uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000
    ```

4.  **Usage with Open-WebUI**:
    *   **Text-to-Speech Engine**: OpenAI
    *   **API Base URL**: `http://YOUR_SERVER_IP:8000/v1`
    *   **API Key**: `not-needed` (or any string)
    *   **TTS Model**: `cosyvoice3`

## Notes
*   The server uses `Fun-CosyVoice3-0.5B` model.
*   It includes text cleaning and normalization.

## VRAM Optimization

To reduce GPU memory usage, see:
- **[QUANTIZATION_QUICK_START.md](QUANTIZATION_QUICK_START.md)** - Quick implementation guide
- **[QUANTIZATION_ANALYSIS.md](QUANTIZATION_ANALYSIS.md)** - Comprehensive analysis

Using 8-bit quantization can reduce VRAM usage from ~2GB to ~1GB with minimal quality impact.
