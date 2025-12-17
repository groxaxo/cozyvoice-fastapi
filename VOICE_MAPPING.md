# Voice Mapping Configuration

This document describes the voice mapping system for CosyVoice FastAPI server.

## Overview

The server now supports **language-specific voice mapping** with automatic voice selection based on:
1. Voice name (e.g., "spanish", "english", "alloy")
2. Language code (e.g., "es", "en", "fr")
3. Language-prefixed voice names (e.g., "es_female", "en-US")

## Voice Samples Directory

Voice samples are stored in: `voice_samples/`

### Available Voices

| Voice File | Voice Name | Language | Description |
|------------|------------|----------|-------------|
| `default.wav` | default | English | Default voice for all requests |
| `multilingual.wav` | multilingual | Multi | Cross-lingual voice |
| `es.wav` | es, spanish | Spanish | Spanish language voice |
| `en.wav` | en, english, alloy, echo, fable, onyx, nova, shimmer | English | English language voice |
| `fr.wav` | fr, french | French | French language voice |
| `it.wav` | it, italian | Italian | Italian language voice |
| `pt.wav` | pt, portuguese | Portuguese | Portuguese language voice |
| `de.wav` | de, german | German | German language voice |
| `ja.wav` | ja, japanese | Japanese | Japanese language voice |
| `ko.wav` | ko, korean | Korean | Korean language voice |
| `zh.wav` | zh, chinese | Chinese | Chinese language voice |

## System Prompts by Language

Each language has a specific system prompt that instructs the model to speak with a native accent from the capital city:

| Language | Capital City | System Prompt |
|----------|--------------|---------------|
| Spanish (es) | Buenos Aires | "Habla como una mujer de Buenos Aires, Argentina, con voz simpática y expresiva." |
| English (en) | London | "Speak like a friendly woman from London, with an expressive voice." |
| French (fr) | Paris | "Parle comme une femme de Paris, avec une voix chaleureuse et expressive." |
| Italian (it) | Rome | "Parla come una donna di Roma, con una voce calda ed espressiva." |
| Portuguese (pt) | Lisbon | "Fale como uma mulher de Lisboa, com voz simpática e expressiva." |
| German (de) | Berlin | "Sprich wie eine Frau aus Berlin, mit einer freundlichen und ausdrucksstarken Stimme." |
| Japanese (ja) | Tokyo | "東京出身の女性のように、明るく表現豊かな声で話してください。" |
| Korean (ko) | Seoul | "서울 출신 여성처럼 밝고 표현력 있는 목소리로 말하세요." |
| Chinese (zh) | Beijing | "请用北京女性的语气，说话亲切而富有表现力。" |

### Base Instruction

All prompts start with: `"You are a professional voice actor."`

The final prompt format is:
```
You are a professional voice actor. [Language-specific instruction]<|endofprompt|>
```

## Voice Selection Logic

The server uses the following priority for voice selection:

1. **Direct Match**: If the voice name exactly matches a file in `voice_samples/` (e.g., "spanish" → "spanish.wav")
2. **Language Code Match**: If the voice name contains a language code (e.g., "es_female" → "es.wav")
3. **Voice-to-Language Map**: Uses `VOICE_LANGUAGE_MAP` to resolve OpenAI-compatible voice names (e.g., "alloy" → "en.wav")
4. **Fallback**: Falls back to "multilingual.wav" or "default.wav"

## Usage Examples

### Example 1: Using Language Code
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer not-needed" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Hola, ¿cómo estás?",
    "voice": "es",
    "response_format": "wav"
  }' \
  --output spanish_output.wav
```
- **Voice File Used**: `voice_samples/es.wav`
- **System Prompt**: "You are a professional voice actor. Habla como una mujer de Buenos Aires, Argentina, con voz simpática y expresiva.<|endofprompt|>"

### Example 2: Using Language Name
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer not-needed" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Bonjour, comment allez-vous?",
    "voice": "french",
    "response_format": "mp3"
  }' \
  --output french_output.mp3
```
- **Voice File Used**: `voice_samples/fr.wav`
- **System Prompt**: "You are a professional voice actor. Parle comme une femme de Paris, avec une voix chaleureuse et expressive.<|endofprompt|>"

### Example 3: Using OpenAI-Compatible Voice Name
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer not-needed" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Hello, how are you today?",
    "voice": "alloy",
    "response_format": "wav"
  }' \
  --output english_output.wav
```
- **Voice File Used**: `voice_samples/en.wav`
- **System Prompt**: "You are a professional voice actor. Speak like a friendly woman from London, with an expressive voice.<|endofprompt|>"

### Example 4: Using Language-Prefixed Voice
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer not-needed" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Ciao, come stai?",
    "voice": "it_female",
    "response_format": "wav"
  }' \
  --output italian_output.wav
```
- **Voice File Used**: `voice_samples/it.wav`
- **System Prompt**: "You are a professional voice actor. Parla come una donna di Roma, con una voce calda ed espressiva.<|endofprompt|>"

## Adding Custom Voices

To add a new voice:

1. Place your voice sample file (WAV, MP3, or FLAC) in the `voice_samples/` directory
2. Name it appropriately (e.g., `aimee.wav`, `maria.wav`)
3. The voice will be automatically discovered and available using the filename (without extension)

Example:
```bash
# Add a custom voice
cp /path/to/my_custom_voice.wav voice_samples/aimee.wav

# Use it in API request
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer not-needed" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Hello from Aimee!",
    "voice": "aimee",
    "response_format": "wav"
  }' \
  --output aimee_output.wav
```

## Integration with Open-WebUI

When configuring Open-WebUI:

1. **Text-to-Speech Engine**: OpenAI
2. **API Base URL**: `http://YOUR_SERVER_IP:8000/v1`
3. **API Key**: `not-needed`
4. **TTS Model**: `cosyvoice3`
5. **Voice**: Choose from:
   - Language codes: `es`, `en`, `fr`, `it`, `pt`, `de`, `ja`, `ko`, `zh`
   - Language names: `spanish`, `english`, `french`, `italian`, `portuguese`, `german`, `japanese`, `korean`, `chinese`
   - OpenAI names: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`
   - Custom voices: Any filename in `voice_samples/` directory

## Technical Details

### Voice Discovery Function
```python
def discover_voice_samples():
    """Discover available voice samples from the voice_samples directory."""
    voice_map = {}
    if os.path.exists(VOICE_SAMPLES_DIR):
        for filename in os.listdir(VOICE_SAMPLES_DIR):
            if filename.endswith(('.wav', '.mp3', '.flac')):
                voice_name = os.path.splitext(filename)[0]
                voice_map[voice_name] = os.path.join(VOICE_SAMPLES_DIR, filename)
    return voice_map
```

### Voice File Resolution
```python
def get_voice_file(voice: Optional[str]) -> str:
    """Get the voice file path for a given voice name."""
    # 1. Discover available voices
    # 2. Direct match check
    # 3. Language code detection
    # 4. VOICE_LANGUAGE_MAP lookup
    # 5. Fallback to multilingual/default
```

### System Prompt Building
```python
def build_prompt_text(voice: Optional[str]) -> str:
    """Build the system prompt based on voice/language."""
    # 1. Start with base instruction
    # 2. Detect language from voice parameter
    # 3. Add language-specific instruction
    # 4. Append <|endofprompt|> token
```

## Notes

- Voice samples are currently duplicates of the base prompts for demonstration
- You can replace any voice file with custom recordings (3+ seconds recommended)
- The system supports zero-shot voice cloning from the reference audio
- All voice files should be clear, high-quality recordings for best results
