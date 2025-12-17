# Voice Samples Directory

This directory contains **voice sample files** used for zero-shot voice cloning with CosyVoice3.

## 🚨 Important Note

**Voice files are NOT included in the Git repository** as they are personal/custom files. Each user should add their own voice samples.

---

## 📁 Adding Your Voice Samples

### Quick Start

1. **Create or obtain voice samples** (WAV, MP3, or FLAC format)
2. **Place them in this directory** (`voice_samples/`)
3. **Name them appropriately** using this convention:
   ```
   [name]-[language_code].[extension]
   ```
   Examples:
   - `aimee-en.wav` (English female voice named Aimee)
   - `lucho-es.wav` (Spanish male voice named Lucho)
   - `marie-fr.mp3` (French female voice named Marie)

### Voice Sample Requirements

✅ **Format**: WAV (recommended), MP3, or FLAC  
✅ **Duration**: 3-10 seconds (optimal for voice cloning)  
✅ **Quality**: Clear, high-quality audio  
✅ **Content**: Natural speech (not singing or shouting)  
✅ **Background**: Minimal or no background noise  

❌ **Avoid**:
- Samples longer than 10 seconds (may cause errors)
- Low-quality or noisy recordings
- Multiple speakers in one sample
- Music or sound effects

---

## 🎭 Recommended Voice Structure

To recreate the full setup, add voices for each language:

### Spanish Voices
```
voice_samples/
├── lucho-es.wav          # Male Spanish voice
├── facundito-es.wav      # Male Spanish voice (alternative)
├── brenda-es.wav         # Female Spanish voice
└── es.wav                # Generic Spanish voice
```

### English Voices
```
voice_samples/
├── aimee-en.wav          # Female English voice
├── michael-en.wav        # Male English voice
└── en.wav                # Generic English voice
```

### Other Languages (Optional)
```
voice_samples/
├── fr.wav                # French
├── it.wav                # Italian
├── de.wav                # German
├── pt.wav                # Portuguese
├── ja.wav                # Japanese
├── ko.wav                # Korean
├── zh.wav                # Chinese
└── multilingual.wav      # Multi-language voice
```

---

## 🎤 Where to Get Voice Samples

### Option 1: Record Your Own
Use any audio recording software (Audacity, Audition, phone app):
1. Record 5-10 seconds of natural speech
2. Export as WAV format (recommended)
3. Save to this directory

### Option 2: Use TTS Services
Generate samples from existing TTS services:
- OpenAI TTS API
- ElevenLabs
- Google Cloud TTS
- Microsoft Azure TTS

### Option 3: Extract from Audio/Video
Extract voice segments from:
- Podcast clips
- YouTube videos (with permission)
- Audiobooks
- Personal recordings

⚠️ **Legal Note**: Ensure you have rights to use any voice samples. Don't use copyrighted content without permission.

---

## 🔧 Usage After Adding Voices

Once you've added voice samples, they'll automatically be discovered by the server:

```bash
# List all available voices
curl http://localhost:8000/v1/voices

# Use a specific voice
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Hello from my custom voice!",
    "voice": "aimee-en"
  }' -o output.wav
```

---

## 📊 Voice Naming Convention

Follow this pattern for automatic language detection:

| Pattern | Example | Language Detected |
|---------|---------|-------------------|
| `name-LANG.ext` | `john-en.wav` | English |
| `name-LANG.ext` | `maria-es.mp3` | Spanish |
| `LANG.ext` | `fr.wav` | French (generic) |
| `name.ext` | `custom.wav` | Multilingual/Default |

### Language Codes
- `en` - English
- `es` - Spanish
- `fr` - French
- `it` - Italian
- `pt` - Portuguese
- `de` - German
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese

---

## 🧪 Testing Your Voices

After adding voices, test them:

```bash
# Test the voice exists
curl http://localhost:8000/v1/voices | jq '.data[] | select(.id=="your-voice-name")'

# Generate test audio
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Testing my new voice sample",
    "voice": "your-voice-name"
  }' -o test.wav

# Play the audio
play test.wav  # or aplay test.wav on Linux
```

---

## 💡 Pro Tips

1. **Use WAV format** for best compatibility and quality
2. **Keep samples short** (3-10 seconds optimal)
3. **Clear speech** - avoid mumbling or whispering
4. **Consistent volume** - normalize audio levels
5. **Natural tone** - conversational speech works best
6. **Test multiple samples** - try different takes to find the best one

---

## 🔄 Current Setup

Check how many voices you have:

```bash
ls -lh voice_samples/*.{wav,mp3,flac} 2>/dev/null | wc -l
```

Restart the server after adding new voices (auto-discovery happens on startup):

```bash
# Using the autonomous launcher
./run_cosyvoice_autonomous.sh

# Or manually
pkill -f openai_tts_cosyvoice_server
conda run -n cosyvoice3 uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000
```

---

## 📚 Documentation

For more details, see:
- [Main README](../README.md) - Complete server documentation
- [VOICE_MAPPING.md](../VOICE_MAPPING.md) - Technical voice mapping details

---

**Happy voice cloning!** 🎉
