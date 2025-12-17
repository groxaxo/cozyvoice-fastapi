import requests
import json
import os
import time

# Configuration
TTS_API_URL = "http://localhost:8000/v1/audio/speech"
TTS_API_KEY = "test-key"
WHISPER_API_URL = "http://100.85.200.52:8887/v1/audio/transcriptions"
OUTPUT_FILE = "test_spanish.wav"

# Spanish text to synthesize
TEXT = "Hola, esto es una prueba de síntesis de voz en español usando CosyVoice tres. Espero que la transcripción sea correcta."

def generate_audio():
    print(f"Generating audio for: '{TEXT}'")
    headers = {
        "Authorization": f"Bearer {TTS_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "cosyvoice3",
        "input": TEXT,
        "voice": "default",
        "response_format": "wav",
        "speed": 1.0
    }
    
    try:
        response = requests.post(TTS_API_URL, headers=headers, json=data, stream=True)
        response.raise_for_status()
        
        with open(OUTPUT_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Audio saved to {OUTPUT_FILE}")
        return True
    except Exception as e:
        print(f"Error generating audio: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return False

def transcribe_audio():
    print(f"Transcribing audio using Whisper at {WHISPER_API_URL}...")
    
    if not os.path.exists(OUTPUT_FILE):
        print("Audio file not found.")
        return None
        
    try:
        with open(OUTPUT_FILE, "rb") as f:
            files = {
                "file": (OUTPUT_FILE, f, "audio/wav")
            }
            data = {
                "model": "whisper-1", # Usually required but often ignored by local servers
                "language": "es"
            }
            # Note: No auth header mentioned for Whisper, but standard is usually Bearer something if protected.
            # Assuming open or internal IP.
            response = requests.post(WHISPER_API_URL, files=files, data=data)
            response.raise_for_status()
            
            result = response.json()
            return result.get("text", "")
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return None

def main():
    # 1. Generate Audio
    if not generate_audio():
        return
    
    # 2. Transcribe Audio
    transcription = transcribe_audio()
    
    if transcription:
        print("\n--- Results ---")
        print(f"Original Text:      {TEXT}")
        print(f"Transcribed Text:   {transcription}")
        
        # Simple correlation check (case insensitive)
        # We check if key words are present
        original_words = set(re.findall(r'\w+', TEXT.lower()))
        transcribed_words = set(re.findall(r'\w+', transcription.lower()))
        
        common = original_words.intersection(transcribed_words)
        overlap = len(common) / len(original_words)
        
        print(f"\nWord Overlap: {overlap:.2%}")
        
        if overlap > 0.8:
            print("SUCCESS: Transcription correlates well with original text.")
        else:
            print("WARNING: Transcription might not match perfectly.")
            
import re
if __name__ == "__main__":
    main()
