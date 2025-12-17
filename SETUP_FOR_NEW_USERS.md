# 🚀 Setup Guide for New Users

This guide is for users who have freshly cloned this repository.

---

## ⚠️ Important: Voice Samples Required

**Voice sample files are NOT included in this repository.** You must add your own voice files before the server will work properly.

---

## 📋 Quick Setup Checklist

### 1. ✅ Clone the Repository
```bash
git clone <repository-url>
cd cozyvoice_fastapi
```

### 2. ✅ Create/Activate Conda Environment
```bash
# Create environment (if not exists)
conda create -n cosyvoice3 python=3.10 -y

# Activate environment
conda activate cosyvoice3

# Install compatible torchvision
pip install torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 3. ⚠️ **ADD VOICE SAMPLES** (Required!)

This is the **most important step** - the server won't work without voices!

**Option A: Record Your Own**
```bash
cd voice_samples/

# Record 3-10 seconds of natural speech
# Save as WAV files with naming convention: name-languagecode.wav
# Examples:
#   - john-en.wav (English male)
#   - maria-es.wav (Spanish female)
#   - pierre-fr.wav (French male)
```

**Option B: Use Existing TTS Services**
```bash
# Generate samples using OpenAI, ElevenLabs, etc.
# Then copy them to voice_samples/ directory
cp /path/to/your/voice/files/*.wav voice_samples/
```

**Option C: Extract from Audio/Video**
```bash
# Extract voice clips from podcasts, videos, etc.
# Use tools like ffmpeg, Audacity, etc.
ffmpeg -i input.mp4 -ss 00:00:10 -t 5 -ar 22050 voice_samples/custom-en.wav
```

**Quick Test - Add at Least One Voice**:
```bash
# For testing, create a simple voice file or download one
# Minimum: Add ONE voice file to get started
# Example structure:
voice_samples/
├── en.wav          # English (required for basic testing)
└── README.md       # This is already there
```

📖 **See [voice_samples/README.md](voice_samples/README.md) for detailed instructions.**

### 4. ✅ Verify Voice Samples
```bash
# Check if you have voice files
ls -lh voice_samples/*.{wav,mp3,flac} 2>/dev/null

# You should see at least one voice file
# If not, go back to step 3!
```

### 5. ✅ Launch the Server
```bash
# Using the auto-restart launcher (recommended)
./run_cosyvoice_autonomous.sh

# Or manually:
export TTS_API_KEY="not-needed"
conda run -n cosyvoice3 uvicorn openai_tts_cosyvoice_server:app --host 0.0.0.0 --port 8000
```

### 6. ✅ Test the Server
```bash
# Check health
curl http://localhost:8000/health

# List voices (should show your added voices)
curl http://localhost:8000/v1/voices

# Generate test audio
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosyvoice3",
    "input": "Hello, this is a test",
    "voice": "en"
  }' -o test.wav

# Play the audio
play test.wav  # or: aplay test.wav
```

---

## 🎯 Recommended Voice Setup

For a complete setup matching the original configuration, add these voices:

### Minimum Setup (Get Started)
```
voice_samples/
├── en.wav          # English voice (any valid voice clip)
└── es.wav          # Spanish voice (optional but recommended)
```

### Recommended Setup (Multi-language)
```
voice_samples/
├── aimee-en.wav    # English female
├── michael-en.wav  # English male
├── lucho-es.wav    # Spanish male
├── brenda-es.wav   # Spanish female
├── fr.wav          # French
├── de.wav          # German
├── it.wav          # Italian
├── pt.wav          # Portuguese
└── multilingual.wav # Fallback voice
```

### Full Setup (Production-Ready)
See [voice_samples/README.md](voice_samples/README.md) for the complete list of 32+ voices.

---

## ❓ Troubleshooting

### "No voices found" Error
**Problem**: Server starts but no voices available  
**Solution**: Add at least one voice file to `voice_samples/` directory

### "Voice file not found" Error
**Problem**: Requested voice doesn't exist  
**Solution**: 
```bash
# List available voices
curl http://localhost:8000/v1/voices

# Use one of the listed voice IDs
```

### "Server won't start" Error
**Problem**: Conda environment issues  
**Solution**:
```bash
# Recreate environment
conda deactivate
conda remove -n cosyvoice3 --all -y
conda create -n cosyvoice3 python=3.10 -y
conda activate cosyvoice3
pip install -r requirements.txt  # If requirements.txt exists
```

### "CUDA out of memory" Error
**Problem**: GPU memory insufficient  
**Solution**: Use smaller model or reduce batch size in code

---

## 📚 Documentation

- **[README.md](README.md)** - Complete server documentation
- **[voice_samples/README.md](voice_samples/README.md)** - Voice setup guide
- **[VOICE_MAPPING.md](VOICE_MAPPING.md)** - Technical voice mapping details
- **[cosyvoice_workflow.md](cosyvoice_workflow.md)** - System architecture

---

## 🎉 You're Ready!

Once you've added voice samples, your CosyVoice FastAPI server is ready to use!

**Next Steps**:
- Integrate with Open-WebUI (see README.md)
- Add more custom voices
- Fine-tune system prompts
- Deploy to production

---

**Need Help?** See the main [README.md](README.md) for comprehensive documentation.
