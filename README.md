# CosyVoice FastAPI Server

A production-ready OpenAI-compatible Text-to-Speech (TTS) server powered by CosyVoice3, offering high-quality multilingual voice synthesis with zero-shot voice cloning capabilities.

## 🌟 Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI TTS API
- **Multi-Language Support**: 9+ languages including English, Spanish, French, German, Italian, Portuguese, Japanese, Korean, and Chinese
- **Zero-Shot Voice Cloning**: Clone any voice from a 3-10 second sample
- **Multiple Audio Formats**: WAV, MP3, FLAC, Opus, AAC
- **Speed Control**: Adjust speech rate
- **Streaming Support**: Low-latency streaming responses
- **Multiple Backends**: Standard PyTorch, vLLM acceleration, and TensorRT support
- **VRAM Optimization**: Optional quantization support to reduce memory usage

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Voice Management](#-voice-management)
- [Integration Examples](#-integration-examples)
- [Performance Optimization](#-performance-optimization)
- [Troubleshooting](#-troubleshooting)
- [Advanced Topics](#-advanced-topics)

## 🚀 Quick Start

### Prerequisites

- Linux system with NVIDIA GPU (CUDA support)
- Conda package manager
- 2-4GB VRAM (depending on configuration)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd cozyvoice-fastapi
   ```

2. **Create and activate Conda environment**:
   ```bash
   conda create -n cosyvoice3 python=3.10 -y
   conda activate cosyvoice3
   pip install torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Add voice samples** (Required!):
   ```bash
   # Add at least one voice file to get started
   # Place WAV/MP3/FLAC files in voice_samples/ directory
   # Example: voice_samples/en.wav for English voice
   ```
   
   See [Voice Management](#-voice-management) for detailed instructions.

4. **Start the server**:
   ```bash
   export TTS_API_KEY="not-needed"
   uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000
   ```

5. **Test the server**:
   ```bash
   curl http://localhost:8000/health
   curl -X POST http://localhost:8000/v1/audio/speech \
     -H "Content-Type: application/json" \
     -d '{"model": "cosyvoice3", "input": "Hello world", "voice": "en"}' \
     -o test.wav
   ```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_API_KEY` | `not-needed` | API authentication key |
| `COSYVOICE_USE_VLLM` | `false` | Enable vLLM acceleration |
| `COSYVOICE_USE_TRT` | `false` | Enable TensorRT acceleration |
| `COSYVOICE_FP16` | `false` | Use FP16 precision |
| `QUANTIZATION_ENABLED` | `false` | Enable BitsAndBytes quantization |
| `QUANTIZATION_BITS` | `4` | Quantization precision (4 or 8 bits) |

### Backend Options

#### Standard PyTorch (Default)
```bash
export TTS_API_KEY="not-needed"
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000
```

#### vLLM Acceleration
```bash
export COSYVOICE_USE_VLLM="true"
./run_cosyvoice_vllm.sh
```

#### Auto-Restart Mode
```bash
./run_cosyvoice_autonomous.sh
```

## 📡 API Reference

### Base URL
```
http://localhost:8000/v1
```

### Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model": "cosyvoice3",
  "backend": "pytorch"
}
```

#### `GET /v1/models`
List available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {"id": "cosyvoice3", "object": "model"}
  ]
}
```

#### `GET /v1/voices`
List available voices.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "en",
      "language": "en",
      "gender": "female"
    }
  ]
}
```

#### `POST /v1/audio/speech`
Generate speech from text.

**Request Body:**
```json
{
  "model": "cosyvoice3",
  "input": "Hello, how are you?",
  "voice": "en",
  "response_format": "wav",
  "speed": 1.0,
  "stream": false,
  "normalization_options": {
    "normalize": true,
    "unit_normalization": true
  }
}
```

**Parameters:**
- `model` (string, required): Must be "cosyvoice3"
- `input` (string, required): Text to convert to speech
- `voice` (string, optional): Voice name or language code (default: "en")
- `response_format` (string, optional): Audio format - "wav", "mp3", "flac", "opus", "aac" (default: "wav")
- `speed` (float, optional): Speech rate multiplier, 0.5-2.0 (default: 1.0)
- `stream` (boolean, optional): Enable streaming response (default: false)
- `normalization_options` (object, optional): Text normalization settings

**Response:**
Binary audio data with appropriate MIME type.

**Example:**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer not-needed" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "The quick brown fox jumps over the lazy dog.",
    "voice": "en",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output speech.mp3
```

#### `POST /v1/warmup`
Pre-warm the model for faster first request.

**Response:**
```json
{
  "status": "warmed",
  "model": "cosyvoice3",
  "backend": "pytorch",
  "sample_rate": 25000,
  "audio_duration_ms": 480
}
```

## 🎭 Voice Management

### Voice Sample Requirements

✅ **Format**: WAV (recommended), MP3, or FLAC  
✅ **Duration**: 3-10 seconds (optimal for voice cloning)  
✅ **Quality**: Clear, high-quality audio  
✅ **Content**: Natural speech (not singing or shouting)  
✅ **Background**: Minimal or no background noise

### Adding Voice Samples

1. **Create or obtain voice samples** in supported formats
2. **Place them in `voice_samples/` directory**
3. **Name using convention**: `[name]-[language_code].[ext]`

**Examples:**
```
voice_samples/
├── en.wav              # Generic English voice
├── es.wav              # Generic Spanish voice
├── aimee-en.wav        # Named English voice
├── lucho-es.wav        # Named Spanish voice
├── fr.wav              # French voice
└── multilingual.wav    # Multi-language voice
```

### Voice Naming Convention

| Pattern | Example | Language Detected |
|---------|---------|-------------------|
| `LANG.ext` | `en.wav` | English (generic) |
| `name-LANG.ext` | `john-en.wav` | English |
| `name-LANG.ext` | `maria-es.mp3` | Spanish |
| `name.ext` | `custom.wav` | Multilingual/Default |

### Supported Language Codes

- `en` - English
- `es` - Spanish
- `fr` - French
- `it` - Italian
- `pt` - Portuguese
- `de` - German
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese

### Voice Selection Logic

The server selects voices using this priority:

1. **Direct Match**: Exact voice name match (e.g., "aimee" → "aimee.wav")
2. **Language Code**: Voice name contains language code (e.g., "es_female" → "es.wav")
3. **OpenAI Compatibility**: Maps OpenAI voice names (e.g., "alloy" → "en.wav")
4. **Fallback**: Uses default voice or CosyVoice base prompt

### OpenAI-Compatible Voice Names

The following OpenAI voice names are automatically mapped:
- `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` → English voices

### Testing Voices

```bash
# List available voices
curl http://localhost:8000/v1/voices

# Test a specific voice
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Testing my voice",
    "voice": "your-voice-name"
  }' -o test.wav
```

## 🔗 Integration Examples

### Open-WebUI Integration

1. Open Open-WebUI settings
2. Navigate to **Text-to-Speech** settings
3. Configure:
   - **Text-to-Speech Engine**: OpenAI
   - **API Base URL**: `http://your-server-ip:8000/v1`
   - **API Key**: `not-needed` (or your configured key)
   - **TTS Model**: `cosyvoice3`
   - **Voice**: Choose from available voices (e.g., `en`, `es`, `alloy`)

### Python Client

```python
import requests

def generate_speech(text, voice="en", output_file="output.wav"):
    url = "http://localhost:8000/v1/audio/speech"
    headers = {
        "Authorization": "Bearer not-needed",
        "Content-Type": "application/json"
    }
    data = {
        "model": "cosyvoice3",
        "input": text,
        "voice": voice,
        "response_format": "wav"
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    with open(output_file, "wb") as f:
        f.write(response.content)
    
    print(f"Audio saved to {output_file}")

# Example usage
generate_speech("Hello, this is a test.", voice="en")
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');
const fs = require('fs');

async function generateSpeech(text, voice = 'en', outputFile = 'output.wav') {
    const url = 'http://localhost:8000/v1/audio/speech';
    
    const response = await axios.post(url, {
        model: 'cosyvoice3',
        input: text,
        voice: voice,
        response_format: 'wav'
    }, {
        headers: {
            'Authorization': 'Bearer not-needed',
            'Content-Type': 'application/json'
        },
        responseType: 'arraybuffer'
    });
    
    fs.writeFileSync(outputFile, response.data);
    console.log(`Audio saved to ${outputFile}`);
}

// Example usage
generateSpeech('Hello, this is a test.', 'en');
```

### cURL Examples

**Basic Usage:**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "cosyvoice3", "input": "Hello world", "voice": "en"}' \
  -o output.wav
```

**Spanish Voice:**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "cosyvoice3", "input": "Hola mundo", "voice": "es"}' \
  -o spanish.wav
```

**MP3 Output with Speed Control:**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Faster speech example",
    "voice": "en",
    "response_format": "mp3",
    "speed": 1.2
  }' \
  -o fast.mp3
```

## ⚡ Performance Optimization

### VRAM Optimization with Quantization

Reduce GPU memory usage by 50-75% using BitsAndBytes quantization. The CosyVoice3 model's LLM component (~0.5B parameters) is the primary target for quantization.

#### Installation

```bash
conda activate cosyvoice3
pip install bitsandbytes>=0.41.0 transformers>=4.30.0 accelerate>=0.20.0
```

#### 8-bit Quantization (Recommended)

**VRAM Reduction**: ~50% (2GB → 1GB)  
**Quality Impact**: Minimal (<3%)

**Usage:**
```bash
export QUANTIZATION_ENABLED="true"
export QUANTIZATION_BITS="8"
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000
```

#### 4-bit Quantization (Maximum Reduction) ⭐ NEW

**VRAM Reduction**: ~75% (2GB → 500MB)  
**Quality Impact**: Small (<5%)  
**Recommended for**: Low-VRAM GPUs or running multiple instances

**Usage:**
```bash
export QUANTIZATION_ENABLED="true"
export QUANTIZATION_BITS="4"
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000
```

#### How It Works

The server now has built-in support for BitsAndBytes quantization:

1. **Automatic Configuration**: Set environment variables to enable quantization
2. **NF4 4-bit**: Uses NormalFloat4 quantization for best quality/size trade-off
3. **Double Quantization**: Further reduces memory with minimal quality loss
4. **Smart Fallback**: Automatically falls back to standard precision if quantization fails
5. **Status Reporting**: Check quantization status via `/health` endpoint

**Environment Variables:**

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `QUANTIZATION_ENABLED` | `true`, `false` | `false` | Enable/disable quantization |
| `QUANTIZATION_BITS` | `4`, `8` | `4` | Quantization precision |

**Checking Status:**

```bash
curl http://localhost:8000/health
# Returns: {"status": "ok", "model": "cosyvoice3", "backend": "pytorch", "quantization": "4-bit"}
```

**Important Notes:**

- ⚠️ Quantization is not compatible with vLLM or TensorRT backends
- ⚠️ Requires NVIDIA GPU with CUDA support
- ⚠️ May require CosyVoice's AutoModel class to support `quantization_config` parameter
- ℹ️ If direct quantization support is unavailable, the server will attempt post-load quantization
- ℹ️ First inference may be slightly slower due to quantization setup

### vLLM Acceleration

For production workloads requiring higher throughput:

1. **Create vLLM environment**:
   ```bash
   conda create -n cosyvoice3_vllm --clone cosyvoice3
   conda activate cosyvoice3_vllm
   pip install vllm==v0.9.0
   ```

2. **Launch with vLLM**:
   ```bash
   export COSYVOICE_USE_VLLM="true"
   ./run_cosyvoice_vllm.sh
   ```

**Benefits**:
- 2-4x higher throughput
- Better GPU utilization
- Optimized for batch processing

### TensorRT Acceleration

For maximum performance on NVIDIA GPUs:

```bash
export COSYVOICE_USE_TRT="true"
export COSYVOICE_FP16="true"
uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000
```

**Benefits**:
- 3-5x faster inference
- Lower latency
- Optimized kernel fusion

## 🔧 Troubleshooting

### Common Issues

#### "No voices found" Error
**Problem**: Server starts but no voices available  
**Solution**: Add at least one voice file to `voice_samples/` directory

```bash
# Check if voices exist
ls -lh voice_samples/*.{wav,mp3,flac} 2>/dev/null

# Add a voice file
cp your_voice.wav voice_samples/en.wav
```

#### "Voice file not found" Error
**Problem**: Requested voice doesn't exist  
**Solution**: List available voices and use one of them

```bash
curl http://localhost:8000/v1/voices
```

#### "CUDA out of memory" Error
**Problem**: GPU memory insufficient  
**Solution**: Enable quantization or use smaller batch size

```bash
export QUANTIZATION_ENABLED="true"
export QUANTIZATION_BITS="8"
```

#### "vLLM not available" Error
**Problem**: vLLM not installed  
**Solution**: Install vLLM or disable it

```bash
# Install vLLM
pip install vllm==v0.9.0

# Or disable vLLM
export COSYVOICE_USE_VLLM="false"
```

#### Server Won't Start
**Problem**: Port already in use or conda environment issues  
**Solution**: Check port and environment

```bash
# Check port usage
lsof -i :8000

# Kill process on port
kill -9 $(lsof -t -i:8000)

# Verify environment
conda env list
conda activate cosyvoice3
```

### Logging

The server uses Python's logging module. To enable debug logging:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

Or set log level via environment:
```bash
export LOG_LEVEL=DEBUG
uvicorn openai_tts_cosyvoice_server:app --log-level debug
```

## 🏗️ Advanced Topics

### System Architecture

The CosyVoice3 model consists of:

1. **LLM Component** (Qwen2LM-based)
   - Converts text to semantic tokens
   - ~0.5B parameters
   - Primary quantization target

2. **Flow Matching Acoustic Model**
   - Converts semantic tokens to acoustic features
   - Masked diffusion architecture

3. **HiFi-GAN Vocoder**
   - Converts acoustic features to waveform
   - High-quality neural vocoder

4. **Text Processing Frontend**
   - Normalizes text
   - Handles markdown, HTML, units
   - Multi-language tokenization

### Text Processing

The server automatically:
- Removes markdown formatting and HTML tags
- Normalizes Unicode characters
- Converts units to spoken form (e.g., "10KB" → "10 kilobytes")
- Splits long text into chunks (max 350 characters)

To disable normalization:
```json
{
  "model": "cosyvoice3",
  "input": "Your text here",
  "normalization_options": {
    "normalize": false,
    "unit_normalization": false
  }
}
```

### Security Considerations

1. **API Key**: Set a strong API key in production
   ```bash
   export TTS_API_KEY="your-secure-key-here"
   ```

2. **Logging**: Sensitive information is not logged by default
   - Model paths are sanitized
   - Voice selections are logged at DEBUG level only
   - No user input is logged

3. **Network**: Use HTTPS in production with reverse proxy (nginx/caddy)

### Production Deployment

#### Using systemd

Create `/etc/systemd/system/cosyvoice-tts.service`:
```ini
[Unit]
Description=CosyVoice TTS Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/cozyvoice-fastapi
Environment="TTS_API_KEY=your-api-key"
Environment="PATH=/home/your-user/miniconda3/envs/cosyvoice3/bin:/usr/bin"
ExecStart=/home/your-user/miniconda3/envs/cosyvoice3/bin/uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable cosyvoice-tts
sudo systemctl start cosyvoice-tts
sudo systemctl status cosyvoice-tts
```

#### Using Docker

Create `Dockerfile`:
```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy application
WORKDIR /app
COPY . .

# Install Python packages
RUN pip3 install -r requirements.txt

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "openai_tts_cosyvoice_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t cosyvoice-tts .
docker run --gpus all -p 8000:8000 -v ./voice_samples:/app/voice_samples cosyvoice-tts
```

#### Using nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name tts.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeout for long audio generation
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

## 📄 License

Please refer to the CosyVoice project license for usage terms.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📞 Support

For issues and questions:
- Check the [Troubleshooting](#-troubleshooting) section
- Review the [API Reference](#-api-reference)
- Open an issue on GitHub

## 🙏 Acknowledgments

- CosyVoice team for the excellent TTS model
- FastAPI for the web framework
- HuggingFace for transformers and BitsAndBytes

---

**Powered by CosyVoice3 • FastAPI • Multi-Language TTS with Zero-Shot Voice Cloning**
