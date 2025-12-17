# CosyVoice FastAPI Project Workflow

```mermaid
flowchart TD
    subgraph "Client Layer"
        A[OpenAI-Compatible Client<br/>e.g., Open-WebUI, curl]
        B[HTTP Request<br/>POST /v1/audio/speech]
    end

    subgraph "FastAPI Server Layer"
        C[FastAPI Server<br/>openai_tts_cosyvoice_server.py]
        D[API Authentication<br/>Bearer Token Check]
        E[Request Validation<br/>SpeechRequest Model]
    end

    subgraph "Text Processing Layer"
        F[Text Cleaning & Normalization]
        G[Markdown/HTML Removal]
        H[Unit Normalization<br/>e.g., 10KB → 10 kilobytes]
        I[Text Splitting<br/>max 350 chars per chunk]
    end

    subgraph "CosyVoice Model Layer"
        J[AutoModel<br/>Fun-CosyVoice3-0.5B]
        K[Zero-Shot Inference<br/>prompt_text + prompt_wav]
        L[Text-to-Speech Generation<br/>LLM-based TTS]
    end

    subgraph "Audio Processing Layer"
        M[Audio Generation<br/>Float32 numpy array]
        N[Sample Rate: 25kHz]
        O[Audio Concatenation<br/>if multiple chunks]
    end

    subgraph "Output Formatting"
        P[Audio Encoding<br/>WAV/MP3/FLAC/Opus/AAC]
        Q[FFmpeg Transcoding<br/>for non-WAV formats]
        R[HTTP Response<br/>with audio payload]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    F --> H
    F --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    M --> O
    O --> P
    P --> Q
    Q --> R
    R --> A

    subgraph "Model Components"
        S[Tokenizer<br/>multilingual support]
        T[Transformer Architecture<br/>encoder/decoder]
        U[Flow Matching<br/>audio generation]
        V[HiFi-GAN<br/>vocoder]
    end

    J --> S
    J --> T
    J --> U
    J --> V

    subgraph "Key Features"
        W[Multilingual Support<br/>9 languages + 18 Chinese dialects]
        X[Zero-Shot Voice Cloning<br/>from 3s reference audio]
        Y[Text Normalization<br/>numbers, symbols, units]
        Z[Streaming Support<br/>150ms latency]
    end

    L --> W
    L --> X
    F --> Y
    C --> Z
```

## Component Details

### 1. **Client Layer**
- **OpenAI-Compatible API**: Uses same endpoints as OpenAI TTS API
- **Authentication**: Bearer token (configurable, default "not-needed")
- **Request Format**: JSON with text input, voice selection, speed control

### 2. **FastAPI Server Layer**
- **Server**: `openai_tts_cosyvoice_server.py`
- **Endpoints**: `/v1/audio/speech`, `/v1/models`
- **Validation**: Pydantic models for request validation

### 3. **Text Processing Layer**
- **Cleaning**: Removes markdown, HTML, control characters
- **Normalization**: Unicode normalization, punctuation standardization
- **Unit Conversion**: Converts technical units to spoken form (e.g., "10KB" → "10 kilobytes")
- **Splitting**: Breaks long text into 350-character chunks

### 4. **CosyVoice Model Layer**
- **Model**: Fun-CosyVoice3-0.5B (0.5 billion parameter LLM-based TTS)
- **Architecture**:
  - **Tokenizer**: Multilingual tokenizer with special control tokens
  - **Transformer**: Encoder-decoder architecture
  - **Flow Matching**: Continuous normalizing flow for audio generation
  - **HiFi-GAN**: Neural vocoder for high-quality audio synthesis
- **Inference Modes**:
  - Zero-shot: Voice cloning from reference audio
  - Cross-lingual: Language switching
  - Instruct: Fine-grained control (speed, emotion, dialect)

### 5. **Audio Processing Layer**
- **Generation**: Produces float32 audio arrays at 25kHz sample rate
- **Concatenation**: Combines multiple audio chunks if text was split
- **Format Conversion**: Supports WAV, MP3, FLAC, Opus, AAC formats

### 6. **Deployment Options**
- **Local**: FastAPI server with GPU acceleration
- **Docker**: Containerized deployment
- **TensorRT-LLM**: NVIDIA acceleration for 4x speedup
- **vLLM**: High-throughput inference engine

## Workflow Sequence

1. **Client Request** → OpenAI-compatible API call
2. **Authentication** → Bearer token validation
3. **Text Processing** → Cleaning, normalization, splitting
4. **Model Inference** → CosyVoice LLM generates audio tokens
5. **Audio Synthesis** → Flow matching + HiFi-GAN produce waveform
6. **Format Encoding** → Convert to requested audio format
7. **Response** → Return audio payload to client

## Key Technologies
- **PyTorch**: Deep learning framework
- **FastAPI**: Web server framework
- **SoundFile**: Audio I/O
- **FFmpeg**: Audio transcoding
- **Modelscope/HuggingFace**: Model distribution
- **CUDA**: GPU acceleration