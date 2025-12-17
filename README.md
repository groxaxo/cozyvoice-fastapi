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
    ```

3.  **Running the Server**:
    ```bash
    export TTS_API_KEY="your-secret-key"
    uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000
    ```

4.  **Usage with Open-WebUI**:
    *   **Text-to-Speech Engine**: OpenAI
    *   **API Base URL**: `http://YOUR_SERVER_IP:8000/v1`
    *   **API Key**: `your-secret-key`
    *   **TTS Model**: `cosyvoice3`

## Notes
*   The server uses `Fun-CosyVoice3-0.5B` model.
*   It includes text cleaning and normalization.
