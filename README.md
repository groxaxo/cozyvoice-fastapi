# 🎙️ CosyVoice FastAPI Server

An **OpenAI-compatible TTS server** powered by CosyVoice3 with **multi-language support**, **intelligent voice mapping**, and **zero-shot voice cloning**.

---

## ✨ Features

- 🎭 **32 Voice Samples** - Male and female voices across multiple languages
- � **9 Languages** - English, Spanish, French, Italian, Portuguese, German, Japanese, Korean, Chinese
- 🔌 **OpenAI Compatible** - Drop-in replacement for OpenAI TTS API
- 🎯 **Smart Voice Selection** - Language codes, voice names, or OpenAI voice names
- 🚀 **Zero-Shot Voice Cloning** - Clone any voice from 3+ seconds of audio
- ⚡ **Fast Inference** - ~150ms latency with streaming support
- 📊 **Automatic Voice Discovery** - Scans `voice_samples/` directory

---

## 🚀 Quick Start

### Prerequisites
- Conda
- NVIDIA GPU with CUDA support

### Installation

```bash
# Activate the environment
conda activate cosyvoice3

# Ensure compatible torchvision version
pip install torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Adding Voice Samples

⚠️ **Important**: Voice samples are **NOT** included in the repository. You need to add your own voice files.

```bash
# Navigate to voice samples directory
cd voice_samples/

# Add your voice samples (WAV, MP3, or FLAC)
# Use naming convention: name-languagecode.extension
# Examples:
#   - aimee-en.wav (English female)
#   - lucho-es.wav (Spanish male)
#   - marie-fr.mp3 (French female)
```

**📖 For detailed instructions**, see [voice_samples/README.md](voice_samples/README.md)

**Quick tips**:
- Use WAV format (recommended)
- 3-10 seconds duration (optimal)
- Clear, high-quality audio
- Minimal background noise

---

## 🚀 Performance Optimization (Optional)

### vLLM Acceleration

For **2-4x faster inference**, enable vLLM acceleration:

#### Prerequisites
```bash
# Create vLLM environment (if not exists)
conda create -n cosyvoice3_vllm --clone cosyvoice3 -y
conda activate cosyvoice3_vllm

# Install vLLM
pip install vllm==v0.9.0 transformers==4.51.3 numpy==1.26.4
```

#### Launch with vLLM
```bash
# Option 1: Use launch script (recommended)
./run_cosyvoice_vllm.sh

# Option 2: Manual launch
export COSYVOICE_USE_VLLM="true"
conda run -n cosyvoice3_vllm uvicorn openai_tts_cosyvoice_server:app \
    --host 0.0.0.0 --port 8001
```

**Performance**: vLLM accelerates the LLM component ~3x, resulting in 2-4x overall speedup.

#### Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `COSYVOICE_USE_VLLM` | `false` | Enable vLLM acceleration |
| `COSYVOICE_USE_TRT` | `false` | Enable TensorRT acceleration |
| `COSYVOICE_FP16` | `false` | Use FP16 precision (CosyVoice3 uses FP32) |

**Note**: vLLM and TensorRT can be enabled simultaneously for maximum performance.

For **batch offline processing** (1000s of files), see [FlashCosyVoice](https://github.com/xingchensong/FlashCosyVoice) which achieves 9x speedup through distributed processing.

---

### Running the Server

```bash
# Standard server (port 8000)
./run_cosyvoice_autonomous.sh

# vLLM-accelerated server (port 8001)
./run_cosyvoice_vllm.sh

# Or manually:
export TTS_API_KEY="not-needed"
conda run -n cosyvoice3 uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000` (or `8001` for vLLM)

---

## 📖 API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. **GET /health** - Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model": "cosyvoice3"
}
```

#### 2. **GET /v1/models** - List Models
```bash
curl http://localhost:8000/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {"id": "cosyvoice3", "object": "model"}
  ]
}
```

#### 3. **GET /v1/voices** - List Available Voices

Returns all 32 available voices with metadata (language, gender, file size).

```bash
curl http://localhost:8000/v1/voices
```

**Response:**
```json
{
  "object": "list",
  "total": 32,
  "data": [
    {
      "id": "lucho-es",
      "language": "es",
      "gender": "male",
      "file_path": "voice_samples/lucho-es.wav",
      "file_size": 191488
    },
    {
      "id": "aimee-en",
      "language": "en",
      "gender": "female",
      "file_path": "voice_samples/aimee-en.wav",
      "file_size": 148375
    }
    // ... 30 more voices
  ]
}
```

**Filter Examples:**
```bash
# Filter Spanish voices
curl -s http://localhost:8000/v1/voices | jq '.data[] | select(.language=="es")'

# Filter female voices
curl -s http://localhost:8000/v1/voices | jq '.data[] | select(.gender=="female")'
```

#### 4. **POST /v1/audio/speech** - Generate Speech

Generate speech from text using specified voice.

**Request Body:**
```json
{
  "model": "cosyvoice3",
  "input": "Text to synthesize",
  "voice": "aimee-en",           // Optional: Voice ID from /v1/voices
  "response_format": "wav",      // Optional: "wav", "mp3", "opus", "aac", "flac"
  "speed": 1.0                   // Optional: Speed multiplier
}
```

**Examples:**

**English (Female):**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Hello, I am Aimee from London",
    "voice": "aimee-en",
    "response_format": "wav"
  }' -o output.wav
```

**Spanish (Male):**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Hola, soy Lucho de Buenos Aires",
    "voice": "lucho-es",
    "response_format": "wav"
  }' -o output.wav
```

**Using Language Codes:**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Bonjour, comment allez-vous?",
    "voice": "fr",
    "response_format": "mp3"
  }' -o french.mp3
```

**Using OpenAI Voice Names:**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Testing with Alloy voice",
    "voice": "alloy",
    "response_format": "wav"
  }' -o alloy.wav
```

---

## 🎭 Available Voices

### Voice Categories (32 Total)

#### Spanish Voices (13)
**Male (5):**
- `lucho-es` ⭐ Perfect - Argentine male
- `facundito-es` ✅ Excellent - Argentine male  
- `faculiado-es` ✅ Good - Argentine male
- `facu-es` - Argentine male
- `facunormal-es` - Argentine male

**Female (8):**
- `brenda-es` ✅ Excellent - Spanish female
- `colombiana-es` - Colombian female
- `es` - Generic Spanish female
- And 5 more variants

#### English Voices (6)
**Male (1):**
- `michael-en` ✅ Excellent - British male

**Female (5):**
- `aimee-en` ⭐ Perfect - British female
- `en` - Generic English female
- And 3 more variants

#### Other Languages (13)
- `de`, `fr`, `it`, `ja`, `ko`, `pt`, `zh` - Generic language voices
- `multilingual` - Cross-lingual voice
- Plus base language WAV files

### Voice Mapping

The server intelligently maps voice requests:

| Voice Name | Maps To | Language |
|------------|---------|----------|
| `en`, `english`, `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` | English voices | English |
| `es`, `spanish` | Spanish voices | Spanish |
| `fr`, `french` | French voices | French |
| `it`, `italian` | Italian voices | Italian |
| `pt`, `portuguese` | Portuguese voices | Portuguese |
| `de`, `german` | German voices | German |
| `ja`, `japanese` | Japanese voices | Japanese |
| `ko`, `korean` | Korean voices | Korean |
| `zh`, `chinese` | Chinese voices | Chinese |

---

## 🔧 Integration

### Open-WebUI Configuration

1. **Text-to-Speech Engine**: OpenAI
2. **API Base URL**: `http://YOUR_SERVER_IP:8000/v1`
3. **API Key**: `not-needed` (or any string)
4. **TTS Model**: `cosyvoice3`
5. **Voice**: Choose from:
   - Language codes: `es`, `en`, `fr`, etc.
   - Specific voices: `lucho-es`, `aimee-en`, etc.
   - OpenAI names: `alloy`, `nova`, `shimmer`, etc.

### Python Example

```python
import requests

# List all voices
response = requests.get("http://localhost:8000/v1/voices")
voices = response.json()
print(f"Total voices: {voices['total']}")

# Generate Spanish speech
response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "cosyvoice3",
        "input": "Hola, ¿cómo estás?",
        "voice": "lucho-es",
        "response_format": "wav"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

---

## 🎨 System Prompts

Each language uses **enhanced system prompts** that specify voice characteristics for authentic accents:

| Language | Accent | Characteristics |
|----------|--------|-----------------|
| **English** | London | Female, posh, charismatic, highly expressive |
| **Spanish** | Buenos Aires | Femenina, elegante, carismática, muy expresiva |
| **French** | Paris | Féminine, élégante, charismatique, très expressive |
| **Italian** | Rome | Femminile, elegante, carismatica, molto espressiva |
| **Portuguese** | Lisbon | Feminina, elegante, carismática, muito expressiva |
| **German** | Berlin | Weiblich, elegant, charismatisch, sehr ausdrucksstark |
| **Japanese** | Tokyo | 女性的、上品、カリスマ的、非常に表現豊か |
| **Korean** | Seoul | 여성스럽고, 우아하며, 카리스마 있고, 표현력 풍부 |
| **Chinese** | Beijing | 女性化、优雅、有魅力、非常有表现力 |

**Example System Prompt (English):**
```
You are a professional voice actor. You are a woman from London, England. 
Your voice is female, posh, charismatic, and highly expressive. 
Speak with refined British elegance and captivating warmth, like a professional British actress.
```

---

## 🛠️ Adding Custom Voices

1. **Place your voice sample** (WAV, MP3, or FLAC) in `voice_samples/` directory
2. **Name it appropriately** (e.g., `aimee-en.wav`, `maria-es.wav`)
3. **The voice is automatically discovered** and available using the filename (without extension)

**Example:**
```bash
# Add a custom voice
cp /path/to/custom_voice.wav voice_samples/custom-en.wav

# Use it immediately
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Hello from custom voice!",
    "voice": "custom-en"
  }' -o custom.wav
```

⚠️ **Best Practices:**
- Voice samples should be **3-10 seconds** long
- Use **clear, high-quality** recordings
- **WAV format** recommended for best results
- Longer samples (>10 seconds) may cause errors

---

## 📊 Technical Details

### Model Information
- **Model**: `Fun-CosyVoice3-0.5B-2512`
- **Architecture**: LLM-based TTS with Flow Matching
- **Sample Rate**: 25kHz
- **Latency**: ~150ms (with streaming)
- **Parameters**: 0.5 billion

### Supported Features
- ✅ Zero-shot voice cloning from reference audio
- ✅ Cross-lingual voice synthesis
- ✅ Multilingual support (9 languages + dialects)
- ✅ Text normalization and cleaning
- ✅ Multiple audio formats (WAV, MP3, FLAC, Opus, AAC)
- ✅ Speed control
- ✅ Streaming support

### Audio Processing
- Text cleaning and normalization
- Markdown/HTML removal
- Unit normalization (e.g., "10KB" → "10 kilobytes")
- Text splitting (max 350 chars per chunk)
- Audio concatenation for long texts

---

## 🧪 Testing

### Test the Server

```bash
# Test English voice
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"cosyvoice3","input":"Hello world","voice":"en"}' \
  -o test.wav

# Test Spanish voice
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"cosyvoice3","input":"Hola mundo","voice":"es"}' \
  -o test_es.wav
```

### Verify with Whisper

```bash
# Generate audio
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"cosyvoice3","input":"Hola mundo","voice":"lucho-es"}' \
  -o test.wav

# Transcribe with Whisper
curl -X POST http://100.85.200.52:8887/v1/audio/transcriptions \
  -F "file=@test.wav" \
  -F "model=whisper-1" \
  -F "language=es"
```

---

## 📚 Documentation

- **[VOICE_MAPPING.md](VOICE_MAPPING.md)** - Comprehensive voice mapping documentation
- **[cosyvoice_workflow.md](cosyvoice_workflow.md)** - System architecture and technical workflow

---

## 🔄 Project Structure

```
cozyvoice_fastapi/
├── openai_tts_cosyvoice_server.py    # Main FastAPI server
├── run_cosyvoice_autonomous.sh       # Auto-restart launcher script
├── detect_voice_languages.py         # Voice language detection utility
├── test_faculiado.py                 # Test script
├── voice_samples/                    # Voice sample directory (32 files)
│   ├── aimee-en.wav
│   ├── lucho-es.wav
│   ├── brenda-es.wav
│   └── ...
├── CosyVoice/                        # CosyVoice model repository
└── tts_tests/                        # Test outputs directory
```

---

## 💡 Usage Tips

1. **Voice Selection Priority**:
   - Direct match: `"aimee-en"` → `voice_samples/aimee-en.wav`
   - Language code: `"es"` → `voice_samples/es.wav`
   - OpenAI names: `"alloy"` → `voice_samples/en.wav`
   - Fallback: `multilingual.wav` or `default.wav`

2. **Response Formats**:
   - `wav` - Lossless, best quality (default)
   - `mp3` - Compressed, smaller size
   - `opus` - Low latency, good for streaming
   - `aac` - Apple ecosystem
   - `flac` - Lossless compression

3. **Performance Tips**:
   - Use WAV format for fastest processing
   - Keep voice samples under 10 seconds
   - Split long texts into sentences for better quality

---

## ⚠️ Known Issues

- Voice samples longer than 10 seconds may cause 500 errors
- Some generated voice files may need trimming for optimal performance
- WAV format recommended over MP3/FLAC for voice samples

---

## 📝 Notes

- The server uses `Fun-CosyVoice3-0.5B-2512` model
- Text cleaning and normalization are automatic
- Voice samples are stored in `voice_samples/` directory
- Each language has a dedicated system prompt for authentic accent and tone
- Zero-shot voice cloning works with 3+ seconds of reference audio

---

## 🤝 Contributing

To add new voices or improve existing ones:
1. Add voice samples to `voice_samples/` directory
2. Follow naming convention: `name-languagecode.wav` (e.g., `john-en.wav`)
3. Ensure sample quality (clear audio, 3-10 seconds)
4. Test with the API

---

---

## 🙏 Credits & Acknowledgments

This project builds upon the excellent work of:

- **[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)** - Base TTS model by FunAudioLLM
- **[FlashCosyVoice](https://github.com/xingchensong/FlashCosyVoice)** - Inspiration for vLLM optimization strategies
- **vLLM Team** - High-performance LLM inference engine

The vLLM integration uses the official CosyVoice vLLM backend for accelerated inference.

---

## 📄 License

This project uses the CosyVoice model. Please refer to the CosyVoice repository for licensing information.

---

**Status**: ✅ Production Ready  
**Server**: 🟢 Running on port 8000  
**Voices**: 32 available  
**Languages**: 9 supported  
**Quality**: Enterprise-grade TTS
